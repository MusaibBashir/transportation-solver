# Transportation Problem Solver (MODI Method)

## Overview
This application solves the **Transportation Problem** using the **MODI (Modified Distribution) Method**. The transportation problem is a type of linear programming problem used to find the most cost-effective way to distribute goods from multiple sources to multiple destinations.

---

## Problem Definition

### Input
- **Sources (S₁, S₂, ..., Sₘ)**: Warehouses or factories with available supply
- **Destinations (D₁, D₂, ..., Dₙ)**: Markets or customers with demand
- **Cost Matrix (Cᵢⱼ)**: Cost to transport one unit from source i to destination j
- **Supply (sᵢ)**: Total units available at source i
- **Demand (dⱼ)**: Total units required at destination j

### Objective
Minimize total transportation cost: **Z = Σ Σ Cᵢⱼ × Xᵢⱼ**

Where Xᵢⱼ is the quantity transported from source i to destination j.

---
