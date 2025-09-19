# test_gene_algo.py
import numpy as np
import pandas as pd
import os
import sys
import random
import string
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from factor.genetic_algo_factor_generator import (
    GeneAlgo,
    IFitnessCalculator,
    ICorrelationCalculator,
)

# -----------------------------------------------------------------------------
# 1. 虚拟数据生成
# -----------------------------------------------------------------------------
def make_fake_market_data(
    n_bars: int = 500,
    n_stocks: int = 50,
    start_date: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    构造一个 MultiIndex DataFrame：
        index = [date, asset]
        columns = open, high, low, close, volume, returns, ...
    这里我们只生成单资产（asset='TEST'），因此 index 只有一层。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_bars, freq="D")

    # 模拟价格随机游走

    # 模拟价格随机游走
    close_vals = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_bars)))
    close = pd.Series(close_vals, index=dates, name="close")  # 关键：转成 Series
    high = close * (1 + np.abs(rng.normal(0, 0.01, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n_bars)))
    open_ = (high + low) / 2 + rng.normal(0, 0.005, n_bars)
    volume = pd.Series(
        rng.integers(1e6, 1e7, n_bars),
        index=dates,
        name="volume",
    )
    returns = close.pct_change()

    df = pd.DataFrame(
        dict(
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            returns=returns,
        ),
        index=dates,
    )
    return df


# -----------------------------------------------------------------------------
# 2. 适配器实现
# -----------------------------------------------------------------------------
class FakeFitness(IFitnessCalculator):
    """
    假装算 IC：因子值与未来 5 日收益的简单相关系数。
    如果因子值全是 NaN，返回 0。
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fwd_ret = data["returns"].shift(-5)

    def calculate_fitness(self, factor: pd.Series) -> float:
        tmp = pd.concat([factor, self.fwd_ret], axis=1).dropna()
        if tmp.empty:
            return 0.0
        corr = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
        return abs(corr) if not np.isnan(corr) else 0.0

    def select_by_fitness(self, fitness_list):
        """
        保留前 50% 的因子（示例用）。
        返回 List[bool]，True 表示保留。
        """
        arr = np.array(fitness_list)
        cutoff = np.percentile(arr, 50)
        return [float(x) >= cutoff for x in fitness_list]


class FakeCorrelation(ICorrelationCalculator):
    """
    皮尔逊相关系数绝对值。
    """

    def calculate_factor_correlation(self, factor1: np.ndarray, factor2: np.ndarray) -> float:
        mask = (~np.isnan(factor1)) & (~np.isnan(factor2))
        if mask.sum() < 10:  # 太少样本直接认为不相关
            return 0.0
        corr = np.corrcoef(factor1[mask], factor2[mask])[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0


# -----------------------------------------------------------------------------
# 3. 单元测试主流程
# -----------------------------------------------------------------------------
def test_gene_algo_e2e():
    data = make_fake_market_data()
    basic_factors = ["open", "high", "low", "close", "volume", "returns"]

    ga = GeneAlgo(
        data=data,
        basic_factors=basic_factors,
        population_size=100, 
        max_depth=4,
        rounds=1,         
        FitnessCalculator=FakeFitness(data),
        correlation_calculator=FakeCorrelation(),
    )

    # 跑完 1 轮初始化 + 变异 + 交叉
    ga.generate_factors()

    # 打印结果
    print(f"After 1 round, population size = {len(ga.all_factors)}")
    for i, expr in enumerate(ga.all_factors[:5]):
        print(f"Expr {i}: {expr}")

    # 断言：至少还有因子存活
    assert len(ga.all_factors) > 0


if __name__ == "__main__":
    test_gene_algo_e2e()