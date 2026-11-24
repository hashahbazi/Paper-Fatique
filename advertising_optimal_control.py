#!/usr/bin/env python3
"""Dual-state advertising fatigue model and steady-policy optimizer."""

from dataclasses import dataclass
import argparse
import math


@dataclass(frozen=True)
class Brand:
    name: str
    baseline_sales: float
    goodwill_efficiency: float
    goodwill_power: float
    goodwill_decay: float
    threshold: float
    fatigue_generation: float
    fatigue_decay: float
    wearout_sensitivity: float
    goodwill_elasticity: float
    buffer_rate: float
    fatigue_elasticity: float
    margin: float = 1.0
    ad_cost: float = 1.0


BRANDS = (
    Brand("Leader", 1000.0, 0.36, 0.85, 0.020, 25.0, 0.18, 0.030, 1.20, 0.65, 0.045, 1.00),
    Brand("Challenger", 700.0, 0.48, 0.80, 0.020, 15.0, 0.32, 0.025, 1.35, 0.60, 0.015, 1.15),
    Brand("Niche", 220.0, 0.24, 0.75, 0.020, 10.0, 0.28, 0.022, 1.50, 0.55, 0.005, 1.25),
)


def softplus(x, smooth=0.1):
    z = x / smooth
    if z > 40.0:
        return x
    if z < -40.0:
        return smooth * math.exp(z)
    return smooth * math.log1p(math.exp(z))


def derivatives(goodwill, annoyance, intensity, brand, smooth):
    dg = brand.goodwill_efficiency * intensity ** brand.goodwill_power - brand.goodwill_decay * goodwill
    da = (brand.fatigue_generation * softplus(intensity - brand.threshold, smooth)
          / (1.0 + brand.buffer_rate * goodwill) - brand.fatigue_decay * annoyance)
    return dg, da


def equilibrium(intensity, brand, horizon=500.0, step=0.1, smooth=0.1):
    goodwill = 0.0
    annoyance = 0.0
    count = int(horizon / step)
    for _ in range(count):
        k1 = derivatives(goodwill, annoyance, intensity, brand, smooth)
        k2 = derivatives(goodwill + step*k1[0]/2, annoyance + step*k1[1]/2, intensity, brand, smooth)
        k3 = derivatives(goodwill + step*k2[0]/2, annoyance + step*k2[1]/2, intensity, brand, smooth)
        k4 = derivatives(goodwill + step*k3[0], annoyance + step*k3[1], intensity, brand, smooth)
        goodwill += step*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6
        annoyance += step*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6
        goodwill = max(0.0, goodwill)
        annoyance = max(0.0, annoyance)
    return goodwill, annoyance


def evaluate(intensity, brand, smooth):
    goodwill, annoyance = equilibrium(intensity, brand, smooth=smooth)
    ratio = annoyance / max(goodwill, 1e-12)
    goodwill_lift = brand.goodwill_elasticity * goodwill / (1.0 + goodwill)
    fatigue_loss = brand.wearout_sensitivity * ratio ** brand.fatigue_elasticity
    sales = max(0.0, brand.baseline_sales * (1.0 + goodwill_lift - fatigue_loss))
    profit = brand.margin * sales - brand.ad_cost * intensity
    return sales, profit, goodwill, annoyance


def optimize(brand, maximum=100.0, resolution=0.25, smooth=0.1):
    n = int(maximum / resolution)
    candidates = [i * resolution for i in range(n + 1)]
    rows = [(u, *evaluate(u, brand, smooth)) for u in candidates]
    best_profit = max(rows, key=lambda row: row[2])
    best_sales = max(rows, key=lambda row: row[1])
    return best_profit, best_sales


def main():
    parser = argparse.ArgumentParser(description="Optimize steady advertising intensity for FMCG brand archetypes.")
    parser.add_argument("--max-intensity", type=float, default=100.0)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--smooth", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_intensity <= 0 or args.resolution <= 0 or args.smooth <= 0:
        parser.error("max-intensity, resolution, and smooth must be positive")
    print("Brand       Profit-opt U   Net profit   Sales-opt U   Sales   Goodwill   Annoyance")
    for brand in BRANDS:
        profit_row, sales_row = optimize(brand, args.max_intensity, args.resolution, args.smooth)
        u, sales, profit, goodwill, annoyance = profit_row
        print(f"{brand.name:<11} {u:>10.2f}   {profit:>10.2f}   {sales_row[0]:>10.2f}   {sales_row[1]:>8.2f}   {goodwill:>8.2f}   {annoyance:>9.2f}")


if __name__ == "__main__":
    main()
