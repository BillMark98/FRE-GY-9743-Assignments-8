"""Valuation dispatch and state-to-market risk conversion from the instructor template."""
from typing import Any
import numpy as np
from fixedincomelib.model import Model
from fixedincomelib.product import Product
from fixedincomelib.valuation.report import PVCashReport, RiskReprt, RiskReport
from fixedincomelib.valuation.valuation_engine import ValuationEngineProduct, ValuationRequest
from fixedincomelib.valuation.valuation_engine_registry import ValuationEngineProductRegistry
from fixedincomelib.valuation.valuation_parameters import ValuationParametersCollection

def create_value_report(
    model : Model,
    product : Product,
    valuation_parameters_collection: ValuationParametersCollection,
    request : ValuationRequest) -> Any:

    engine = ValuationEngineProductRegistry.new_valuation_engine(
        model,
        product,
        valuation_parameters_collection,
        request)

    engine.calculate_value()
    if request in [ValuationRequest.PV_DETAILED, ValuationRequest.PV, ValuationRequest.CASH]:
        v : PVCashReport = engine.get_value_and_cash()
        if request == ValuationRequest.PV_DETAILED:
            return v
        else:
            return v.pv if request == ValuationRequest.PV else v.cash
    elif request == ValuationRequest.FIRST_ORDER_RISK:
        return risk_calculation(engine)
    elif request == ValuationRequest.CASHFLOWS_REPORT:
        return engine.create_cash_flows_report()
    elif request == ValuationRequest.PAR_RATE_OR_SPREAD:
        return engine.par_rate_or_spread()
    else:
        raise Exception(f'Request is not currently supported.')


def risk_calculation(engine : ValuationEngineProduct):
    gradient = []
    engine.calculate_first_order_risk(gradient, 1., False)
    # Preserve the original assignment's direct-IFR report and state_gradient API.
    if not any(component.calibration_product for component in engine.model.components_.values()):
        return RiskReport.from_model_gradient(engine.model, gradient)
    # dV/dX^M = dV/dX^I \cdot J^{-1}, where J is the model jacobian dX^M/dX^I
    # step 1: flatten gradient
    gradient_flat = np.concatenate(gradient, axis=0) # dim : 1 x len(X^I)
    # step 2: get model (lazy) jacobian
    engine.model.calculate_model_jacobian()
    # HW8 task 4: change variables from internal state X^I to market quotes X^M.
    # The calibration defines X^M as a function of X^I with Jacobian
    # J = dX^M/dX^I (square: one quote per state variable), so by the chain
    # rule the ROW gradients satisfy
    #   grad_{X^M} V = grad_{X^I} V . J^{-1}     <=>     J^T x = grad_{X^I} V^T.
    # Solving the transposed system gives the same numbers as multiplying by
    # the explicit inverse, in one factorisation and without forming J^{-1}.
    # Units stay raw (value per 1.0 = 10,000 bp of quote); the report's
    # display() applies each quote's UNIT (1e-4) to show value per +1 bp.
    risk = np.linalg.solve(engine.model.model_jacobian.T, gradient_flat)
    # step 5: make a nice report
    risk_report = RiskReprt(engine.model.risk_postprocess(risk))
    return risk_report
