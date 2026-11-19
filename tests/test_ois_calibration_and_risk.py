"""Regression tests for Assignment 8 (SOFR OIS calibration, quote risk, hedging).

Covers the five HW8 tasks:

  2a. ValuationEngineProductRfrSwap.calculate_value   fixed-leg annuity
  2b. ValuationEngineProductRfrSwap.calculate_value   OIS par rate
  2c. ValuationEngineProductRfrSwap.grad_at_par       d(par rate) / d(internal state)
  3.  YieldCurve.calculate_model_jacobian             J = dX^M / dX^I, one row per quote
  4.  valuation.utilities.risk_calculation            grad_{X^M} V = grad_{X^I} V . J^{-1}

The notebook ships the instructor's outputs for the PV, the risk column and the
hedge notionals, and those are asserted here.  Reference numbers alone would not
say WHY they are right, so each task is also checked against something that
does not depend on the implementation:

  * the annuity and par rate are recomputed from the cashflow report;
  * the par rate does not depend on coupon, notional or direction, and a swap
    struck at its par rate is worth zero;
  * the par-rate gradient and the Jacobian match central finite differences of
    the internal state;
  * the quote risk matches "bump one market quote and RE-CALIBRATE the curve",
    which exercises 2c, 3 and 4 together and is the definition of quote risk;
  * a par swap has risk only to its own quote, equal to -N * annuity per unit
    rate, so the hedge matrix is diagonal;
  * the hedged book is flat to first order under re-calibrated rate scenarios.

Run either way (about a minute: every bump re-bootstraps a curve)::

    python tests/test_ois_calibration_and_risk.py
    pytest tests/test_ois_calibration_and_risk.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents
             if (p / "fixedincomelib" / "yield_curve" / "calibration_utils.py").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fixedincomelib import (  # noqa: E402
    qfCreateData1D, qfCreateDataCollection, qfCreateDataGeneric, qfCreateBuildMethod,
    qfCreateModelBuildMethodCollection, qfCreateModel, qfCreateValuationParameters,
    qfCreateValuationParametersCollection, qfCreateProductFromDataConvention,
    qfCreateValueReport,
)
from fixedincomelib.date import Date  # noqa: E402
from fixedincomelib.valuation import ValuationEngineProductRegistry, ValuationRequest  # noqa: E402
from fixedincomelib.valuation.report import RiskReport  # noqa: E402
from fixedincomelib.yield_curve.model_builder import YieldCurveBuilder  # noqa: E402

# ----------------------------------------------------------------------------
# The notebook's inputs
# ----------------------------------------------------------------------------
QUOTES = (pd.read_excel(_ROOT / "fixedincomelib/data/SOFR OIS.xlsx", index_col="Date", engine="openpyxl").T
          / 10_000)
VALUE_DATE = pd.Timestamp(QUOTES.columns[0]).date().isoformat()
QUOTES.columns = ["values"]
TENORS = list(QUOTES.index)
N_TARGET = 1_000_000
BP = 1e-4
SOFR, FUNDING = "SOFR-1B", "SOFR-1B-FLAT"

# Instructor outputs saved in the supplied notebook (cells 8 and 10).
REFERENCE_PV = -24829.03
REFERENCE_RISK = [1.336135, 0.382009, 0.597304, 1.038225, 0.823435, 1.766412, 1.566545, 3.531423,
                  -355.339457, -401.432157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
REFERENCE_OWN_QUOTE_RISK = [-97.770067, -189.912728, -277.730519, -361.778939, -441.907361, -519.132398,
                            -591.670793, -660.730533, -726.424834, -789.177720, -905.487777, -1057.514068,
                            -1264.029757, -1425.446493, -1554.292696, -1747.511351, -1886.178011]
REFERENCE_HEDGE_NOTIONAL = [13666.089735, 2011.495956, 2150.660087, 2869.776317, 1863.365811, 3402.622933,
                            2647.663054, 5344.724260, -489162.044569, -508671.426672,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

VP = qfCreateValuationParametersCollection([
    qfCreateValuationParameters("FUNDING INDEX PARAMETER", {"FUNDING INDEX": FUNDING})])

_MODELS = {}


def build_model(quotes=None, cache=True):
    """Calibrate SOFR to OIS quotes exactly as the notebook does (zero funding spread)."""
    quotes = QUOTES if quotes is None else quotes
    key = (tuple(quotes.index), tuple(float(v) for v in quotes["values"]))
    if cache and key in _MODELS:
        return _MODELS[key]
    funding_table = qfCreateDataGeneric("DATA GENERIC", "USD-FUNDING", pd.DataFrame([
        ["OVERNIGHT INDEX SWAP", "USD-SOFR-OIS", FUNDING],
        ["SPREAD ZERO RATE", "SOFR-1B-FLAT-OVER-SOFR-1B-ZERO-SPREAD", FUNDING],
    ], columns=["DATA TYPE", "DATA CONVENTION", "FUNDING IDENTIFIER"]))
    build_methods = qfCreateModelBuildMethodCollection([
        qfCreateBuildMethod("YIELD_CURVE_COMMON", {"TARGET": "USD", "FUNDING PARAMETERS": "USD-FUNDING",
                                                   "SOLVER": "BRENTQ"}),
        qfCreateBuildMethod("YIELD_CURVE_INDEX", {
            "TARGET": SOFR, "OVERNIGHT INDEX SWAP": "USD-SOFR-OIS",
            "INTERPOLATION METHOD": "PIECEWISE_CONSTANT_LEFT_CONTINUOUS", "EXTRAPOLATION METHOD": "FLAT"}),
        qfCreateBuildMethod("YIELD_CURVE_FUNDING", {
            "TARGET": FUNDING, "SPREAD ZERO RATE": "SOFR-1B-FLAT-OVER-SOFR-1B-ZERO-SPREAD",
            "INTERPOLATION METHOD": "PIECEWISE_CONSTANT_LEFT_CONTINUOUS", "EXTRAPOLATION METHOD": "FLAT"}),
    ])
    data = qfCreateDataCollection([
        qfCreateData1D("OVERNIGHT INDEX SWAP", "USD-SOFR-OIS", quotes),
        qfCreateData1D("SPREAD ZERO RATE", "SOFR-1B-FLAT-OVER-SOFR-1B-ZERO-SPREAD",
                       pd.DataFrame({"values": [0.0]}, index=["1Y"])),
        funding_table,
    ])
    model = qfCreateModel(VALUE_DATE, "YIELD_CURVE", data, build_methods)
    if cache:
        _MODELS[key] = model
    return model


def bumped(quotes, shifts):
    """Copy of ``quotes`` with ``shifts`` (tenor -> decimal shift, or one array) added."""
    out = quotes.copy()
    if isinstance(shifts, dict):
        for tenor, shift in shifts.items():
            out.loc[tenor, "values"] += shift
    else:
        out["values"] = out["values"].to_numpy() + np.asarray(shifts, dtype=float)
    return out


def ois(axis1, rate, notional=N_TARGET, pay_or_rec="receive"):
    return qfCreateProductFromDataConvention(VALUE_DATE, "USD-SOFR-OIS", axis1, rate,
                                             notional=notional, pay_or_rec=pay_or_rec)


def target_swap():
    return ois("9Y6M", 0.045)


def pv(model, product):
    return qfCreateValueReport(model, product, VP, "PV")[0][1]


def engine_for(model, product):
    engine = ValuationEngineProductRegistry.new_valuation_engine(model, product, VP, ValuationRequest.PV)
    engine.calculate_value()
    return engine


def quote_risk(model, product):
    """Displayed risk (value per +1 bp): (funding-spread entry, the OIS entries)."""
    frame = qfCreateValueReport(model, product, VP, "FirstOrderRisk").display()
    values = frame["VALUES"].to_numpy(dtype=float)
    is_ois = (frame["DATA_TYPE"] == "OVERNIGHT INDEX SWAP").to_numpy()
    return values[~is_ois], values[is_ois]


def state_layout(model):
    """(component name, node index) for every state variable, in flat gradient order."""
    layout = []
    for name in model.component_indices:            # dict order == block order
        layout += [(name, i) for i in range(model.retrieve_model_component(name).num_state_data)]
    return layout


def state_fd(model, func, step=1e-6):
    """Central finite differences of ``func()`` with respect to every state variable."""
    rows = []
    for name, node in state_layout(model):
        model.perturb_model_parameter(name, node, step)
        up = np.atleast_1d(func()).astype(float)
        model.perturb_model_parameter(name, node, -2.0 * step)
        down = np.atleast_1d(func()).astype(float)
        model.perturb_model_parameter(name, node, step)          # restore
        rows.append((up - down) / (2.0 * step))
    return np.array(rows).T          # (len(func()), n_state)


# ----------------------------------------------------------------------------
# Calibration (supplied bootstrap; needs 2a/2b to run at all)
# ----------------------------------------------------------------------------
def test_bootstrap_reprices_every_quote_and_pins_nodes_at_last_payment_dates():
    model = build_model()
    sofr = model.retrieve_model_component(SOFR)
    assert [row[2] for row in sofr.market_data] == TENORS
    par = np.array([qfCreateValueReport(model, s, VP, "ParRateOrSpread") for s in sofr.calibration_product])
    assert np.abs(par - QUOTES["values"].to_numpy()).max() < 1e-12            # < 1e-8 bp
    for swap, node_time in zip(sofr.calibration_product, sofr.state_data[0]):
        assert abs(pv(model, swap)) < 1e-6                                     # per 10,000 notional
        # each node sits at its instrument's LAST PAYMENT date (accrual end + 2 business days)
        assert swap.last_date > swap.termination_date
        expected = (swap.last_date.serialNumber() - Date(VALUE_DATE).serialNumber())
        assert abs(node_time * 365.25 - expected) < 1.0                        # ACT/ACT year fraction
    assert np.all(np.diff(sofr.state_data[0]) > 0)


def test_funding_spread_is_zero_so_discounting_is_single_curve():
    model = build_model()
    funding = model.retrieve_model_component(FUNDING)
    # zero up to the root solver's tolerance (Brent leaves ~5e-14, not an exact 0.0)
    assert funding.num_state_data == 1 and abs(funding.state_data[1][0]) < 1e-11
    for date in ("2027-04-13", "2036-04-14", "2070-01-02"):
        assert abs(model.discount_factor(FUNDING, Date(date)) - model.discount_factor(SOFR, Date(date))) < 1e-10
    assert list(model.component_indices.values()) == [0, 1]                    # funding block first, then SOFR
    assert model.retrieve_model_component_index(FUNDING) == 0


# ----------------------------------------------------------------------------
# Tasks 2a / 2b: annuity and par rate
# ----------------------------------------------------------------------------
def test_annuity_and_par_rate_agree_with_the_cashflow_report():
    model = build_model()
    target = target_swap()
    engine = engine_for(model, target)
    flows = qfCreateValueReport(model, target, VP, "CashflowsReport").display()
    fixed, floating = flows[flows["Leg"] == "Fixed"], flows[flows["Leg"] == "Floating"]
    annuity = float((fixed["Accrual"] * fixed["Discount Factor"]).sum())      # sum alpha_i D(pay_i)
    assert abs(engine.annuity_ - annuity) < 1e-14
    floating_pv = -float(floating["PV"].sum())                                 # report signs the paid leg
    par = floating_pv / (N_TARGET * annuity)
    assert abs(engine.par_rate_or_spread() - par) < 1e-15
    assert abs(qfCreateValueReport(model, target, VP, "ParRateOrSpread") - par) < 1e-15
    # the swap is an annuity paying (K - K*):  PV = N * A * (K - K*)
    assert abs(engine.value - N_TARGET * annuity * (0.045 - par)) < 1e-8
    assert abs(engine.value - REFERENCE_PV) < 0.005                            # instructor output, 2 dp


def test_par_rate_is_independent_of_coupon_notional_and_direction():
    model = build_model()
    reference = engine_for(model, target_swap())
    par, annuity = reference.par_rate_or_spread(), reference.annuity_
    variants = [ois("9Y6M", 0.02), ois("9Y6M", 0.09, pay_or_rec="pay"), ois("9Y6M", 0.045, notional=1.0),
                ois("9Y6M", 0.0),                                   # zero coupon: direct-annuity branch
                ois("9Y6M", 0.0, notional=250.0, pay_or_rec="pay")]
    for product in variants:
        engine = engine_for(model, product)
        assert abs(engine.par_rate_or_spread() - par) < 1e-14
        assert abs(engine.annuity_ - annuity) < 1e-13
    assert abs(pv(model, ois("9Y6M", par))) < 1e-8                  # struck at par -> worth zero
    assert abs(pv(model, ois("9Y6M", 0.045)) + pv(model, ois("9Y6M", 0.045, pay_or_rec="pay"))) < 1e-9
    # a floating spread s raises the par rate by s (same schedule on both legs for this convention)
    spread_swap = qfCreateProductFromDataConvention(VALUE_DATE, "USD-SOFR-OIS", "9Y6M", 0.045,
                                                    notional=N_TARGET, spread=0.001)
    assert abs(engine_for(model, spread_swap).par_rate_or_spread() - par - 0.001) < 1e-12


# ----------------------------------------------------------------------------
# Task 2c: par-rate gradient
# ----------------------------------------------------------------------------
def test_par_rate_gradient_matches_state_finite_differences():
    model = build_model()
    products = [target_swap(), ois("9Y6M", 0.045, pay_or_rec="pay"), ois("9Y6M", 0.0), ois("3Y", 0.01),
                ois("1Yx5Y", 0.03, notional=5e5)]
    for product in products:
        engine = engine_for(model, product)
        blocks = engine.grad_at_par()
        assert [len(b) for b in blocks] == model.num_sub_components           # component order and sizes
        analytic = np.concatenate(blocks)

        def par_rate():
            engine.calculate_value()
            return engine.par_rate_or_spread()

        numeric = state_fd(model, par_rate)[0]
        assert np.abs(analytic - numeric).max() < 1e-9, type(product).__name__
    # the gradient does not depend on coupon or direction either
    grads = [np.concatenate(engine_for(model, p).grad_at_par()) for p in products[:3]]
    assert np.abs(grads[0] - grads[1]).max() < 1e-14 and np.abs(grads[0] - grads[2]).max() < 1e-13


def test_grad_at_par_refreshes_stale_engine_values():
    model = build_model()
    engine = engine_for(model, target_swap())
    before = np.concatenate(engine.grad_at_par())
    model.perturb_model_parameter(SOFR, 3, 0.01)                              # move the curve under the engine
    moved = np.concatenate(engine.grad_at_par())
    fresh = np.concatenate(engine_for(model, target_swap()).grad_at_par())
    model.perturb_model_parameter(SOFR, 3, -0.01)
    assert np.abs(moved - fresh).max() < 1e-15 and np.abs(moved - before).max() > 1e-6


# ----------------------------------------------------------------------------
# Task 3: calibration Jacobian
# ----------------------------------------------------------------------------
def all_calibration_par_rates(model):
    out = []
    for name in model.component_indices:
        component = model.retrieve_model_component(name)
        for product, funding in zip(component.calibration_product, component.calibration_funding):
            vp = qfCreateValuationParametersCollection([
                qfCreateValuationParameters("FUNDING INDEX PARAMETER", {"FUNDING INDEX": funding})])
            out.append(qfCreateValueReport(model, product, vp, "ParRateOrSpread")
                       if name != FUNDING else _implied_zero_spread(model, product))
    return np.array(out, dtype=float)


def _implied_zero_spread(model, product):
    """-ln(D_funding / D_index) / T: the quote the zero-spread instrument calibrates to."""
    engine = ValuationEngineProductRegistry.new_valuation_engine(model, product, VP, ValuationRequest.PV)
    engine.calculate_value()
    return -np.log(engine.df_funding_ / engine.df_base_) / engine.time_to_expiry_


def test_jacobian_matches_finite_differences_and_is_block_lower_triangular():
    model = build_model()
    model.calculate_model_jacobian()
    jacobian = model.model_jacobian
    n = sum(model.num_sub_components)
    assert jacobian.shape == (n, n) == (18, 18)
    numeric = state_fd(model, lambda: all_calibration_par_rates(model))
    assert np.abs(jacobian - numeric).max() < 1e-8
    # bootstrap structure: quote i depends only on nodes <= i ...
    assert np.abs(np.triu(jacobian, 1)).max() < 1e-14
    assert np.all(np.diag(jacobian) > 0.0)
    # ... the funding quote on the funding node alone, with unit sensitivity ...
    assert np.allclose(jacobian[0], np.eye(n)[0], atol=1e-14)
    # ... and each OIS quote is a positive combination of the forwards up to its node.
    # The weights sum to slightly MORE than one: a +1bp parallel move in the
    # continuously-compounded ACT/ACT forwards moves an annually-paid ACT/360
    # par rate by (1 + K*alpha) * tau / alpha ~ 1.03bp (more at 40Y-50Y).
    sofr_block = jacobian[1:, 1:]
    assert np.all(sofr_block[np.tril_indices(17)] > 0.0)
    row_sums = sofr_block.sum(axis=1)
    assert np.all((row_sums > 1.0) & (row_sums < 1.15))
    assert np.linalg.cond(jacobian) < 100.0


def test_jacobian_is_lazy_and_invalidated_by_a_perturbation():
    model = build_model(cache=False)
    assert not model.is_jacobian_calculated and model.model_jacobian.size == 0
    quote_risk(model, target_swap())                                           # first risk request builds it
    assert model.is_jacobian_calculated
    first = model.model_jacobian.copy()
    model.perturb_model_parameter(SOFR, 4, 0.002)
    assert not model.is_jacobian_calculated                                    # stale -> rebuilt on demand
    model.calculate_model_jacobian()
    assert np.abs(model.model_jacobian - first).max() > 1e-5
    model.perturb_model_parameter(SOFR, 4, -0.002)
    model.calculate_model_jacobian()
    assert np.abs(model.model_jacobian - first).max() < 1e-12


# ----------------------------------------------------------------------------
# Task 4: state risk -> quote risk
# ----------------------------------------------------------------------------
def test_quote_risk_is_state_risk_times_inverse_jacobian():
    model = build_model()
    target = target_swap()
    report = qfCreateValueReport(model, target, VP, "FirstOrderRisk")
    raw = np.array([float(row[-1]) for row in report.content])                 # per unit rate, before display
    engine = engine_for(model, target)
    state_gradient = []
    engine.calculate_first_order_risk(state_gradient, 1.0, False)
    flat = np.concatenate(state_gradient)
    numeric_state = state_fd(model, lambda: pv(model, target), step=1e-6)[0]
    assert np.abs(flat - numeric_state).max() < 1e-3 * 1.0                     # dPV/d(state), USD per unit rate
    jacobian = model.model_jacobian
    assert np.abs(raw @ jacobian - flat).max() < 1e-6                          # defining identity
    assert np.abs(raw - flat @ np.linalg.inv(jacobian)).max() < 1e-6           # the notebook's formula
    displayed = report.display()
    assert np.abs(displayed["VALUES"].to_numpy(dtype=float) - raw * BP).max() < 1e-12   # UNIT applied once
    assert list(displayed["AXIS1"]) == ["1Y"] + TENORS
    assert np.allclose(displayed["MARKET_QUOTE"].astype(float).to_numpy()[1:], QUOTES["values"].to_numpy())


def test_notebook_reference_values():
    model = build_model()
    target = target_swap()
    assert abs(pv(model, target) - REFERENCE_PV) < 0.005
    _, risk = quote_risk(model, target)
    assert np.abs(risk - np.array(REFERENCE_RISK)).max() < 5e-7                # printed to 6 dp
    hedges = [ois(tenor, rate) for tenor, rate in QUOTES["values"].items()]
    hedge_risk = np.array([quote_risk(model, hedge)[1] for hedge in hedges])
    assert np.abs(np.diag(hedge_risk) - np.array(REFERENCE_OWN_QUOTE_RISK)).max() < 5e-7
    weights = np.linalg.solve(hedge_risk.T, -risk)
    assert np.abs(weights * N_TARGET - np.array(REFERENCE_HEDGE_NOTIONAL)).max() < 5e-6
    assert np.abs(risk + weights @ hedge_risk).max() < 1e-9                    # residual risk


def test_quote_risk_matches_bump_and_recalibrate():
    """The definition of quote risk: move one quote, rebuild the curve, revalue."""
    target = target_swap()
    step = 1e-5                                                                # 0.1 bp, central
    # every quote of a curve truncated at 12Y (fast to rebuild) ...
    short = QUOTES.loc[:"12Y"]
    _, analytic_short = quote_risk(build_model(short), target)
    for i, tenor in enumerate(short.index):
        up = pv(build_model(bumped(short, {tenor: step}), cache=False), target)
        down = pv(build_model(bumped(short, {tenor: -step}), cache=False), target)
        assert abs((up - down) / (2 * step) * BP - analytic_short[i]) < 1e-4, tenor
    # ... and the two dominant buckets plus one beyond the swap on the full 50Y curve
    _, analytic_full = quote_risk(build_model(), target)
    for tenor in ("9Y", "10Y", "30Y"):
        i = TENORS.index(tenor)
        up = pv(build_model(bumped(QUOTES, {tenor: step}), cache=False), target)
        down = pv(build_model(bumped(QUOTES, {tenor: -step}), cache=False), target)
        assert abs((up - down) / (2 * step) * BP - analytic_full[i]) < 1e-4, tenor


def test_funding_spread_risk_matches_rebootstrap_with_ois_quotes_held():
    """Risk to the funding-spread quote: shift the spread, re-solve SOFR to the SAME OIS quotes."""
    target = target_swap()
    funding_risk, _ = quote_risk(build_model(), target)
    common = [bm for _, bm in build_model().build_method_collection.items
              if type(bm).__name__ == "YieldCurveBuildMethodCommon"][0]
    step, values = 1e-5, []
    for shift in (step, -step):
        model = build_model(cache=False)
        model.perturb_model_parameter(FUNDING, 0, shift, True)                 # funding node == spread quote
        YieldCurveBuilder.calibrate_single_component_from_mkt_data(
            model, model.retrieve_model_component(SOFR), common)
        par = [qfCreateValueReport(model, s, VP, "ParRateOrSpread")
               for s in model.retrieve_model_component(SOFR).calibration_product]
        assert np.abs(np.array(par) - QUOTES["values"].to_numpy()).max() < 1e-11
        values.append(pv(model, target))
    assert abs((values[0] - values[1]) / (2 * step) * BP - funding_risk[0]) < 1e-4
    assert 5.0 < funding_risk[0] < 20.0      # a liability of ~25k discounted ~5y harder: +~12 per bp


def test_risk_is_local_to_the_quotes_spanning_the_swap():
    target = target_swap()
    _, full = quote_risk(build_model(), target)
    assert np.all(full[TENORS.index("12Y"):] == 0.0)                           # exactly zero beyond 10Y
    _, short = quote_risk(build_model(QUOTES.loc[:"12Y"]), target)
    assert np.abs(short - full[:len(short)]).max() < 1e-7                      # longer quotes are irrelevant
    assert full[TENORS.index("9Y")] < -300 and full[TENORS.index("10Y")] < -300
    # receiving 4.5% below par is a par swap plus a PAID annuity; the annuity gains when rates rise
    assert np.all(full[:TENORS.index("9Y")] > 0.0) and full[:TENORS.index("9Y")].sum() < 15.0
    # total risk ~ -N * annuity * 1bp of the target (parallel shift of all quotes)
    annuity = engine_for(build_model(), target).annuity_
    assert abs(full.sum() / (-N_TARGET * annuity * BP) - 1.0) < 0.02


# ----------------------------------------------------------------------------
# Hedging
# ----------------------------------------------------------------------------
def test_par_hedges_have_diagonal_risk_equal_to_minus_notional_times_annuity():
    model = build_model()
    hedges = [ois(tenor, rate) for tenor, rate in QUOTES["values"].items()]
    hedge_risk = np.array([quote_risk(model, hedge)[1] for hedge in hedges])
    annuities = np.array([engine_for(model, hedge).annuity_ for hedge in hedges])
    assert np.abs(np.diag(hedge_risk) + N_TARGET * annuities * BP).max() < 1e-6
    off_diagonal = hedge_risk - np.diag(np.diag(hedge_risk))
    assert np.abs(off_diagonal).max() < 1e-6                                   # a par swap sees only its own quote
    assert all(abs(pv(model, hedge)) < 1e-4 for hedge in hedges)               # and costs nothing


def test_hedged_book_is_flat_to_first_order_under_recalibrated_scenarios():
    base = build_model()
    target = target_swap()
    _, risk = quote_risk(base, target)
    hedges = [ois(tenor, rate) for tenor, rate in QUOTES["values"].items()]
    hedge_risk = np.array([quote_risk(base, hedge)[1] for hedge in hedges])
    weights = np.linalg.solve(hedge_risk.T, -risk)
    book = [(target, 1.0)] + [(hedge, w) for hedge, w in zip(hedges, weights) if w != 0.0]
    years = np.array([float(t[:-1]) for t in TENORS])
    scenarios = {
        "parallel +10bp": np.full(len(TENORS), 10 * BP),
        "steepener (-5bp at 1Y to +15bp at 50Y)": (-5 + 20 * (years - 1) / 49) * BP,
        "9Y/10Y butterfly": np.where(np.isin(TENORS, ["9Y"]), 8 * BP, 0.0) - np.where(np.isin(TENORS, ["10Y"]), 6 * BP, 0.0),
    }
    for name, shifts in scenarios.items():
        pnl = {}
        for scale in (1.0, 0.5):
            shocked = build_model(bumped(QUOTES, scale * shifts), cache=False)
            unhedged = pv(shocked, target) - pv(base, target)
            hedged = sum(w * (pv(shocked, p) - pv(base, p)) for p, w in book)
            pnl[scale] = (unhedged, hedged)
        unhedged, hedged = pnl[1.0]
        predicted = float(risk @ (shifts / BP))                                # first-order P&L from the report
        assert abs(unhedged - predicted) < 0.06 * abs(predicted), name         # the report explains the move
        assert abs(hedged) < 0.06 * abs(unhedged), (name, hedged, unhedged)
        # first-order flat means what is left is second order: half the shock, a quarter of the P&L
        assert 3.8 < hedged / pnl[0.5][1] < 4.2, (name, pnl)
        if not name.startswith("9Y/10Y"):                                      # shocks whose first-order terms do not offset
            assert abs(hedged) < 1e-4 * abs(unhedged), (name, hedged, unhedged)


# ----------------------------------------------------------------------------
# Earlier behaviour that must survive
# ----------------------------------------------------------------------------
def test_direct_ifr_models_keep_their_state_risk_report():
    """Models built from IFR points have no calibration instruments: J is the identity."""
    nodes = ["2027-04-02", "2028-10-02", "2031-01-02"]
    data = qfCreateDataCollection([
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-IFR",
                       pd.DataFrame({"values": [0.04, 0.045, 0.05]}, index=nodes)),
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-FUNDING-IFR",
                       pd.DataFrame({"values": [0.0005]}, index=["2027-01-04"])),
    ])
    build_methods = qfCreateModelBuildMethodCollection([
        qfCreateBuildMethod("YIELD_CURVE_INDEX", {"TARGET": SOFR, "INSTANTANEOUS FORWARD RATE": "SOFR-1B-IFR"}),
        qfCreateBuildMethod("YIELD_CURVE_FUNDING", {"TARGET": FUNDING, "REFERENCE INDEX": SOFR,
                                                    "INSTANTANEOUS FORWARD RATE": "SOFR-1B-FUNDING-IFR"}),
    ])
    model = qfCreateModel(VALUE_DATE, "YIELD_CURVE", data, build_methods)
    swap = ois("3Y", 0.04)
    report = qfCreateValueReport(model, swap, VP, "FirstOrderRisk")
    assert isinstance(report, RiskReport)
    analytic = np.concatenate(report.state_gradient)
    numeric = state_fd(model, lambda: pv(model, swap))[0]
    assert np.abs(analytic - numeric).max() < 1e-3
    model.calculate_model_jacobian()
    assert np.array_equal(model.model_jacobian, np.eye(4))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:                     # assertion OR unexpected error
            failed += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
