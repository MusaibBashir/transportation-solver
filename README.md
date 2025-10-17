# Transportation Problem Solver (MODI Method) - Documentation

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

## Solution Process

### Step 1: Problem Balancing
Compare total supply with total demand
- **Balanced**: Supply = Demand → Proceed to Step 2
- **Unbalanced**: Supply ≠ Demand → Add dummy source or destination with cost = 0

- If Demand > Supply: Add dummy source with supply = (Demand - Supply)
- If Supply > Demand: Add dummy destination with demand = (Supply - Demand)

---

### Step 2: Initial Feasible Solution
Find an initial basic feasible solution using one of three methods:

#### **Method 1: NWCM (North-West Corner Method)**
**Process**:
1. Start from top-left cell (1,1)
2. Allocate minimum of: supply at current row OR demand at current column
3. If supply exhausted → move down one row
4. If demand met → move right one column
5. Continue until all supply/demand is allocated

#### **Method 2: LCM (Least Cost Method)**
**Process**:
1. Find cell with minimum cost in the entire matrix
2. Allocate minimum of: supply at that row OR demand at that column
3. Mark row/column as satisfied if supply/demand is exhausted
4. Repeat with remaining cells until all allocated

#### **Method 3: VAM (Vogel's Approximation Method)**
**Process**:
1. Calculate **penalty** for each row/column:
   - Penalty = (2nd minimum cost) - (minimum cost)
2. Find row/column with maximum penalty
3. In that row/column, allocate to cell with minimum cost
4. Update supply/demand and remove satisfied rows/columns
5. Repeat until all allocated

---

### Step 3: Optimality Check & MODI Method

#### **3A: Check for Degeneracy**
A solution is **degenerate** if: Number of Basic Variables ≠ (m + n - 1)

Where:
- m = number of sources
- n = number of destinations
- Basic variables = non-zero allocations

**If Degenerate**:
- Add epsilon allocation (0.000001) to a cell that maintains independence
- Ensures we can calculate u and v values

#### **3B: Calculate u and v Multipliers**
These represent the shadow prices (dual variables) of the problem.

**Formula**: For all **basic variables (allocated cells)**: uᵢ + vⱼ = Cᵢⱼ

**Algorithm**:
1. Set u₁ = 0 (arbitrary starting point)
2. For each cell where allocation > 0:
   - If uᵢ is known and vⱼ unknown: vⱼ = Cᵢⱼ - uᵢ
   - If vⱼ is known and uᵢ unknown: uᵢ = Cᵢⱼ - vⱼ
3. Repeat until all u and v values are determined

#### **3C: Calculate Reduced Costs (Opportunity Costs)**
For all **non-basic variables (unallocated cells)**:

**Formula**: Δᵢⱼ = uᵢ + vⱼ - Cᵢⱼ

**Interpretation**:
- Δᵢⱼ > 0: Cost would increase if we allocate here (not good)
- Δᵢⱼ = 0: Cost remains same if we allocate here
- Δᵢⱼ < 0: Cost would decrease if we allocate here (good)

#### **3D: Optimality Check**
**Optimal Solution Found When**: All Δᵢⱼ ≤ 0 for non-basic variables

If all reduced costs are non-positive, the current allocation is optimal.

#### **3E: Find Entering Variable (If Not Optimal)**
**Entering Variable**: Cell with **maximum positive reduced cost**

Choose: **max(Δᵢⱼ)** where Δᵢⱼ > 0

This cell will enter the basis (become a basic variable).

---

### Step 4: Stepping Stone Path & Reallocation

#### **4A: Find Stepping Stone Path**
A closed-loop path connecting the entering cell to existing allocations.

**Process**:
1. Use **Independent Allocation Check**:
   - Eliminate rows/columns with fewer than 2 allocations
   - Identify independent set of rows and columns
2. Build path using **Manhattan distance** (closest unvisited cells)
3. Path alternates between:
   - **Even positions (+)**: Increase allocation
   - **Odd positions (-)**: Decrease allocation

**Path Structure**: [Start] → [+] → [-] → [+] → ... → [Start]

#### **4B: Update Allocations**
1. Find minimum allocation in **odd positions** (cells being decreased)
2. Add this minimum to **even positions** (cells being increased)
3. Subtract this minimum from **odd positions** (cells being decreased)
4. Entering cell (position 0) becomes basic with allocation = minimum value

**Result**: One cell leaves the basis, one enters the basis (maintains m+n-1 basics)

---

### Step 5: Iteration
Repeat Steps 3-4 until optimality is achieved.

---
