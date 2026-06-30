"""Tests for product dataclass construction, properties, and field semantics."""

from __future__ import annotations

import numpy as np
import pytest

from foureng.products import (
    AsianOption,
    BarrierOption,
    BasketOption,
    BermudanOption,
    BestOfOption,
    ChooserOption,
    CliquetOption,
    CompoundOption,
    DigitalOption,
    EuropeanOption,
    ExchangeOption,
    ForwardStartOption,
    LookbackOption,
    SpreadOption,
    VarianceOption,
    VarianceSwap,
)

# ── EuropeanOption ─────────────────────────────────────────────────────────


class TestEuropeanOption:
    def test_basic_call(self):
        opt = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
        assert opt.strike == 100.0
        assert opt.maturity == 1.0
        assert opt.cp == 1
        assert opt.product_type == "european"

    def test_basic_put(self):
        opt = EuropeanOption(strike=90.0, maturity=0.5, cp=-1)
        assert opt.cp == -1

    def test_frozen(self):
        opt = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
        with pytest.raises((AttributeError, TypeError)):
            opt.strike = 110.0  # type: ignore[misc]

    def test_negative_strike_rejected(self):
        with pytest.raises(ValueError, match="strike"):
            EuropeanOption(strike=-10.0, maturity=1.0, cp=1)

    def test_zero_strike_rejected(self):
        with pytest.raises(ValueError, match="strike"):
            EuropeanOption(strike=0.0, maturity=1.0, cp=1)

    def test_zero_maturity_rejected(self):
        with pytest.raises(ValueError, match="maturity"):
            EuropeanOption(strike=100.0, maturity=0.0, cp=1)

    def test_invalid_cp_rejected(self):
        with pytest.raises(ValueError, match="cp"):
            EuropeanOption(strike=100.0, maturity=1.0, cp=0)


# ── DigitalOption ──────────────────────────────────────────────────────────


class TestDigitalOption:
    def test_cash_or_nothing_call(self):
        opt = DigitalOption(strike=100.0, maturity=1.0, cp=1, payoff_type="cash_or_nothing")
        assert opt.product_type == "digital"
        assert opt.payoff_type == "cash_or_nothing"
        assert opt.cash_amount == 1.0

    def test_asset_or_nothing_put(self):
        opt = DigitalOption(strike=100.0, maturity=1.0, cp=-1, payoff_type="asset_or_nothing")
        assert opt.payoff_type == "asset_or_nothing"

    def test_invalid_payoff_type(self):
        with pytest.raises(ValueError, match="payoff_type"):
            DigitalOption(strike=100.0, maturity=1.0, cp=1, payoff_type="binary")  # type: ignore

    def test_negative_cash_amount(self):
        with pytest.raises(ValueError, match="cash_amount"):
            DigitalOption(strike=100.0, maturity=1.0, cp=1, cash_amount=-1.0)


# ── BarrierOption ──────────────────────────────────────────────────────────


class TestBarrierOption:
    def test_down_out_call(self):
        opt = BarrierOption(strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_out")
        assert opt.is_knockout
        assert not opt.is_knockin
        assert opt.product_type == "barrier"

    def test_up_in_put(self):
        opt = BarrierOption(strike=100.0, barrier=120.0, maturity=1.0, cp=-1, barrier_type="up_in")
        assert opt.is_knockin

    def test_invalid_barrier_type(self):
        with pytest.raises(ValueError, match="barrier_type"):
            BarrierOption(strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="in_down")  # type: ignore

    def test_down_barrier_above_strike_rejected(self):
        with pytest.raises(ValueError, match="down-barrier"):
            BarrierOption(strike=100.0, barrier=110.0, maturity=1.0, cp=1, barrier_type="down_out")

    def test_up_barrier_below_strike_rejected(self):
        with pytest.raises(ValueError, match="up-barrier"):
            BarrierOption(strike=100.0, barrier=90.0, maturity=1.0, cp=1, barrier_type="up_out")

    def test_rebate_stored(self):
        opt = BarrierOption(
            strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_out", rebate=5.0
        )
        assert opt.rebate == 5.0

    def test_negative_rebate_rejected(self):
        with pytest.raises(ValueError, match="rebate"):
            BarrierOption(
                strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_out", rebate=-1.0
            )


# ── AsianOption ────────────────────────────────────────────────────────────


class TestAsianOption:
    def test_arithmetic_asian(self):
        t = np.linspace(0.25, 1.0, 4)
        opt = AsianOption(
            strike=100.0, maturity=1.0, cp=1, average_type="arithmetic", monitoring_times=t
        )
        assert opt.product_type == "asian"
        assert len(opt.monitoring_times) == 4

    def test_geometric_asian(self):
        t = np.array([0.5, 1.0])
        opt = AsianOption(
            strike=100.0, maturity=1.0, cp=1, average_type="geometric", monitoring_times=t
        )
        assert opt.average_type == "geometric"

    def test_monitoring_times_not_sorted_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            AsianOption(
                strike=100.0, maturity=1.0, cp=1, monitoring_times=np.array([0.8, 0.5, 1.0])
            )

    def test_monitoring_times_exceeds_maturity_rejected(self):
        with pytest.raises(ValueError, match="maturity"):
            AsianOption(strike=100.0, maturity=1.0, cp=1, monitoring_times=np.array([0.5, 1.5]))


# ── BermudanOption ─────────────────────────────────────────────────────────


class TestBermudanOption:
    def test_two_exercise_dates(self):
        t = np.array([0.5, 1.0])
        opt = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=t)
        assert opt.n_exercise == 2
        assert not opt.is_european_equivalent

    def test_single_exercise_at_maturity_is_european(self):
        t = np.array([1.0])
        opt = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=t)
        assert opt.is_european_equivalent

    def test_unsorted_exercise_times_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([1.0, 0.5]))

    def test_exercise_beyond_maturity_rejected(self):
        with pytest.raises(ValueError, match="maturity"):
            BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([0.5, 2.0]))


# ── ForwardStartOption ─────────────────────────────────────────────────────


class TestForwardStartOption:
    def test_basic(self):
        opt = ForwardStartOption(start_time=0.5, maturity=1.0, cp=1, alpha=1.0)
        assert opt.tenor == pytest.approx(0.5)
        assert opt.product_type == "forward_start"

    def test_zero_start_allowed(self):
        opt = ForwardStartOption(start_time=0.0, maturity=1.0, cp=1)
        assert opt.start_time == 0.0

    def test_start_at_maturity_rejected(self):
        with pytest.raises(ValueError, match="maturity"):
            ForwardStartOption(start_time=1.0, maturity=1.0, cp=1)

    def test_start_after_maturity_rejected(self):
        with pytest.raises(ValueError, match="maturity"):
            ForwardStartOption(start_time=2.0, maturity=1.0, cp=1)

    def test_alpha_zero_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            ForwardStartOption(start_time=0.0, maturity=1.0, cp=1, alpha=0.0)


# ── VarianceSwap / VarianceOption ──────────────────────────────────────────


class TestVarianceProducts:
    def test_variance_swap(self):
        t = np.linspace(1 / 12, 1.0, 12)
        vs = VarianceSwap(maturity=1.0, sampling_times=t)
        assert vs.product_type == "variance_swap"

    def test_variance_option_call(self):
        t = np.array([1.0])
        vo = VarianceOption(strike=0.04, maturity=1.0, cp=1, sampling_times=t)
        assert vo.product_type == "variance_option"

    def test_negative_strike_variance_option_rejected(self):
        with pytest.raises(ValueError, match="strike"):
            VarianceOption(strike=-0.01, maturity=1.0, cp=1, sampling_times=np.array([1.0]))


# ── LookbackOption ─────────────────────────────────────────────────────────


class TestLookbackOption:
    def test_floating_call(self):
        opt = LookbackOption(maturity=1.0, cp=1, strike_type="floating")
        assert opt.product_type == "lookback"

    def test_fixed_strike_zero_rejected(self):
        with pytest.raises(ValueError, match="strike"):
            LookbackOption(maturity=1.0, cp=1, strike_type="fixed", strike=0.0)

    def test_fixed_strike_positive_ok(self):
        opt = LookbackOption(maturity=1.0, cp=1, strike_type="fixed", strike=100.0)
        assert opt.strike == 100.0


# ── CliquetOption ──────────────────────────────────────────────────────────


class TestCliquetOption:
    def test_basic(self):
        t = np.array([0.25, 0.5, 0.75, 1.0])
        opt = CliquetOption(maturity=1.0, reset_times=t, local_cap=0.05, local_floor=-0.03)
        assert opt.product_type == "cliquet"
        assert opt.local_cap == 0.05

    def test_cap_below_floor_rejected(self):
        with pytest.raises(ValueError, match="local_cap"):
            CliquetOption(
                maturity=1.0, reset_times=np.array([1.0]), local_cap=-0.1, local_floor=0.0
            )


# ── MultiAsset products ────────────────────────────────────────────────────


class TestMultiAssetProducts:
    def test_exchange_option(self):
        opt = ExchangeOption(maturity=1.0, spot2=95.0, q2=0.01, sigma2=0.25, rho=0.4)
        assert opt.product_type == "exchange"
        assert opt.spot2 == 95.0

    def test_exchange_invalid_rho_rejected(self):
        with pytest.raises(ValueError, match="rho"):
            ExchangeOption(maturity=1.0, rho=1.5)

    def test_basket_equal_weights(self):
        w = np.array([0.5, 0.5])
        opt = BasketOption(strike=100.0, maturity=1.0, cp=1, weights=w)
        assert opt.product_type == "basket"

    def test_basket_rejects_bad_corr_shape(self):
        with pytest.raises(ValueError, match="corr_matrix"):
            BasketOption(
                strike=100.0,
                maturity=1.0,
                cp=1,
                weights=np.array([0.4, 0.6]),
                corr_matrix=np.eye(3),
            )

    def test_basket_zero_total_weight_rejected(self):
        with pytest.raises(ValueError, match="sum to a positive"):
            BasketOption(strike=100.0, maturity=1.0, cp=1, weights=np.array([0.0, 0.0]))

    def test_spread_option_zero_strike_allowed(self):
        opt = SpreadOption(strike=0.0, maturity=1.0, cp=1)
        assert opt.strike == 0.0

    def test_spread_invalid_rho_rejected(self):
        with pytest.raises(ValueError, match="rho"):
            SpreadOption(strike=0.0, maturity=1.0, cp=1, rho=2.0)

    def test_spread_negative_strike_rejected(self):
        with pytest.raises(ValueError, match="strike"):
            SpreadOption(strike=-1.0, maturity=1.0, cp=1)

    def test_best_of_one_asset_rejected(self):
        with pytest.raises(ValueError, match="n_assets"):
            BestOfOption(strike=100.0, maturity=1.0, cp=1, n_assets=1)

    def test_best_of_rejects_nonsymmetric_corr(self):
        with pytest.raises(ValueError, match="symmetric"):
            BestOfOption(
                strike=100.0,
                maturity=1.0,
                cp=1,
                corr_matrix=np.array([[1.0, 0.2], [0.1, 1.0]]),
            )


# ── CompoundOption ─────────────────────────────────────────────────────────


class TestCompoundOption:
    def test_call_on_call(self):
        opt = CompoundOption(
            strike_outer=5.0, strike_inner=100.0,
            maturity_outer=0.5, maturity_inner=1.0,
            cp_outer=1, cp_inner=1,
        )
        assert opt.product_type == "compound"
        assert opt.cp_outer == 1
        assert opt.cp_inner == 1

    def test_put_on_put(self):
        opt = CompoundOption(
            strike_outer=3.0, strike_inner=100.0,
            maturity_outer=0.25, maturity_inner=0.75,
            cp_outer=-1, cp_inner=-1,
        )
        assert opt.cp_outer == -1
        assert opt.cp_inner == -1

    def test_frozen(self):
        opt = CompoundOption(
            strike_outer=5.0, strike_inner=100.0,
            maturity_outer=0.5, maturity_inner=1.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            opt.strike_outer = 10.0  # type: ignore[misc]

    def test_outer_maturity_ge_inner_rejected(self):
        with pytest.raises(ValueError):
            CompoundOption(
                strike_outer=5.0, strike_inner=100.0,
                maturity_outer=1.0, maturity_inner=1.0,
            )

    def test_outer_maturity_after_inner_rejected(self):
        with pytest.raises(ValueError):
            CompoundOption(
                strike_outer=5.0, strike_inner=100.0,
                maturity_outer=1.5, maturity_inner=1.0,
            )

    def test_negative_outer_strike_rejected(self):
        with pytest.raises(ValueError):
            CompoundOption(
                strike_outer=-1.0, strike_inner=100.0,
                maturity_outer=0.5, maturity_inner=1.0,
            )

    def test_zero_outer_strike_allowed(self):
        opt = CompoundOption(
            strike_outer=0.0, strike_inner=100.0,
            maturity_outer=0.5, maturity_inner=1.0,
        )
        assert opt.strike_outer == 0.0

    def test_invalid_cp_outer_rejected(self):
        with pytest.raises(ValueError):
            CompoundOption(
                strike_outer=5.0, strike_inner=100.0,
                maturity_outer=0.5, maturity_inner=1.0,
                cp_outer=0,
            )

    def test_invalid_cp_inner_rejected(self):
        with pytest.raises(ValueError):
            CompoundOption(
                strike_outer=5.0, strike_inner=100.0,
                maturity_outer=0.5, maturity_inner=1.0,
                cp_inner=2,
            )


# ── ChooserOption ──────────────────────────────────────────────────────────


class TestChooserOption:
    def test_basic(self):
        opt = ChooserOption(strike=100.0, maturity_choice=0.5, maturity_expiry=1.0)
        assert opt.product_type == "chooser"
        assert opt.strike == 100.0
        assert opt.maturity_choice == 0.5
        assert opt.maturity_expiry == 1.0

    def test_frozen(self):
        opt = ChooserOption(strike=100.0, maturity_choice=0.5, maturity_expiry=1.0)
        with pytest.raises((AttributeError, TypeError)):
            opt.strike = 110.0  # type: ignore[misc]

    def test_choice_ge_expiry_rejected(self):
        with pytest.raises(ValueError):
            ChooserOption(strike=100.0, maturity_choice=1.0, maturity_expiry=1.0)

    def test_choice_after_expiry_rejected(self):
        with pytest.raises(ValueError):
            ChooserOption(strike=100.0, maturity_choice=1.5, maturity_expiry=1.0)

    def test_zero_choice_time_rejected(self):
        with pytest.raises(ValueError):
            ChooserOption(strike=100.0, maturity_choice=0.0, maturity_expiry=1.0)

    def test_zero_strike_rejected(self):
        with pytest.raises(ValueError):
            ChooserOption(strike=0.0, maturity_choice=0.5, maturity_expiry=1.0)

    def test_negative_strike_rejected(self):
        with pytest.raises(ValueError):
            ChooserOption(strike=-10.0, maturity_choice=0.5, maturity_expiry=1.0)
