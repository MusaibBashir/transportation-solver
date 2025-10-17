import streamlit as st
import pandas as pd
import numpy as np
import sys

def isBalanced(costMatrix):
    supply = sum([row[-1] for row in costMatrix[:-1]])
    demand = sum(costMatrix[-1][:-1])
    return supply == demand, supply, demand

def balanceProblem(costMatrix, supply, demand):
    if demand > supply:
        new_row = [0] * len(costMatrix[0])
        new_row[-1] = demand - supply
        costMatrix.insert(-1, new_row)
    elif supply > demand:
        for i in range(len(costMatrix) - 1):
            costMatrix[i].insert(-1, 0)
        costMatrix[-1].insert(-1, supply - demand)
    return costMatrix

def getTotalCost(costMatrix, method='NWCM'):
  
    if method == 'NWCM':
        return getNWCM(costMatrix)
    elif method == 'LCM':
        return getLCM(costMatrix)
    elif method == 'VAM':
        return getVAM(costMatrix)
    else:
        return getNWCM(costMatrix)

def getNWCM(costMatrix):
    """North-West Corner Method"""
    m = len(costMatrix)
    n = len(costMatrix[0])
    allocMatrix = [[0 for _ in range(n)] for _ in range(m)]
    costMat_copy = [row[:] for row in costMatrix]
    
    numAllocated = 0
    totalCost = 0
    i, j = 0, 0
    
    while i < m - 1 and j < n - 1:
        x = min(costMat_copy[i][-1], costMat_copy[-1][j])
        
        allocMatrix[i][j] = x
        totalCost += x * costMat_copy[i][j]
        
        costMat_copy[i][-1] -= x
        costMat_copy[-1][j] -= x
        numAllocated += 1
        
        if costMat_copy[i][-1] == 0:
            i += 1
        else:
            j += 1
    
    for r in range(m - 1):
        allocMatrix[r][-1] = costMatrix[r][-1]
    for c in range(n):
        allocMatrix[-1][c] = costMatrix[-1][c]
    
    return totalCost, numAllocated, allocMatrix

def getLCM(costMatrix):
    """Least Cost Method"""
    m = len(costMatrix)
    n = len(costMatrix[0])
    allocMatrix = [[0 for _ in range(n)] for _ in range(m)]
    costMat_copy = [row[:] for row in costMatrix]
    
    numAllocated = 0
    totalCost = 0
    
    while True:
        min_cost = float('inf')
        min_i, min_j = -1, -1
        
        for i in range(m - 1):
            for j in range(n - 1):
                if costMat_copy[i][-1] > 0 and costMat_copy[-1][j] > 0:
                    if costMat_copy[i][j] < min_cost:
                        min_cost = costMat_copy[i][j]
                        min_i, min_j = i, j
        
        if min_i == -1:
            break
        
        # Allocate minimum of supply and demand
        x = min(costMat_copy[min_i][-1], costMat_copy[-1][min_j])
        allocMatrix[min_i][min_j] = x
        totalCost += x * costMat_copy[min_i][min_j]
        
        costMat_copy[min_i][-1] -= x
        costMat_copy[-1][min_j] -= x
        numAllocated += 1
    
    for r in range(m - 1):
        allocMatrix[r][-1] = costMatrix[r][-1]
    for c in range(n):
        allocMatrix[-1][c] = costMatrix[-1][c]
    
    return totalCost, numAllocated, allocMatrix

def getVAM(costMatrix):
    """Vogel's Approximation Method"""
    m = len(costMatrix)
    n = len(costMatrix[0])
    allocMatrix = [[0 for _ in range(n)] for _ in range(m)]
    costMat_copy = [row[:] for row in costMatrix]
    
    numAllocated = 0
    totalCost = 0
    
    while True:
        # Calculate penalties for each row
        row_penalties = []
        for i in range(m - 1):
            if costMat_copy[i][-1] > 0:
                costs = sorted([costMat_copy[i][j] for j in range(n - 1) if costMat_copy[-1][j] > 0])
                if len(costs) >= 2:
                    penalty = costs[1] - costs[0]
                elif len(costs) == 1:
                    penalty = costs[0]
                else:
                    penalty = 0
                row_penalties.append((penalty, i, 'row'))
            else:
                row_penalties.append((0, i, 'row'))
        
        # Calculate penalties for each column
        col_penalties = []
        for j in range(n - 1):
            if costMat_copy[-1][j] > 0:
                costs = sorted([costMat_copy[i][j] for i in range(m - 1) if costMat_copy[i][-1] > 0])
                if len(costs) >= 2:
                    penalty = costs[1] - costs[0]
                elif len(costs) == 1:
                    penalty = costs[0]
                else:
                    penalty = 0
                col_penalties.append((penalty, j, 'col'))
            else:
                col_penalties.append((0, j, 'col'))
        
        # Find maximum penalty
        all_penalties = row_penalties + col_penalties
        max_penalty_item = max(all_penalties, key=lambda x: x[0])
        
        if max_penalty_item[0] == 0:
            break
        
        if max_penalty_item[2] == 'row':
            i = max_penalty_item[1]
            min_cost = float('inf')
            min_j = -1
            for j in range(n - 1):
                if costMat_copy[-1][j] > 0 and costMat_copy[i][j] < min_cost:
                    min_cost = costMat_copy[i][j]
                    min_j = j
        else:
            j = max_penalty_item[1]
            min_cost = float('inf')
            min_i = -1
            for i in range(m - 1):
                if costMat_copy[i][-1] > 0 and costMat_copy[i][j] < min_cost:
                    min_cost = costMat_copy[i][j]
                    min_i = i
            i = min_i
        
        x = min(costMat_copy[i][-1], costMat_copy[-1][min_j if max_penalty_item[2] == 'row' else j])
        if max_penalty_item[2] == 'row':
            allocMatrix[i][min_j] = x
            totalCost += x * costMat_copy[i][min_j]
            costMat_copy[i][-1] -= x
            costMat_copy[-1][min_j] -= x
        else:
            allocMatrix[i][j] = x
            totalCost += x * costMat_copy[i][j]
            costMat_copy[i][-1] -= x
            costMat_copy[-1][j] -= x
        
        numAllocated += 1
    
    for r in range(m - 1):
        allocMatrix[r][-1] = costMatrix[r][-1]
    for c in range(n):
        allocMatrix[-1][c] = costMatrix[-1][c]
    
    return totalCost, numAllocated, allocMatrix

def isDegenerate(costMatrix, numAllocated):
    m = len(costMatrix) - 1
    n = len(costMatrix[0]) - 1
    return numAllocated != (m + n - 1)

def removeDeg(allMat, cstMat):
    """Removes degeneracy by adding a small allocation (epsilon)."""
    m, n = len(allMat), len(allMat[0])
    min_cost = float('inf')
    best_pos = None

    for i in range(m):
        for j in range(n):
            if allMat[i][j] == 0:
                if cstMat[i][j] < min_cost:
                    min_cost = cstMat[i][j]
                    best_pos = (i, j)

    if best_pos:
        allMat[best_pos[0]][best_pos[1]] = 1e-6  # Epsilon
    
    return allMat

def findUV(costMatrix, allocMat):
    u = [None] * len(allocMat)
    v = [None] * len(allocMat[0])
    u[0] = 0 
    
    max_iterations = len(allocMat) + len(allocMat[0])
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        found_new = False
        
        for i in range(len(allocMat)):
            for j in range(len(allocMat[0])):
                if allocMat[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = costMatrix[i][j] - u[i]
                        found_new = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = costMatrix[i][j] - v[j]
                        found_new = True
        
        if not found_new:
            break
    
    return u, v

def findDeltas(cstMat, allMat, u, v):
    """Calculates opportunity costs (reduced costs) for unallocated cells. Reduced cost = ui + vj - Cij"""
    deltas = [[None for _ in range(len(allMat[0]))] for _ in range(len(allMat))]
    for i in range(len(allMat)):
        for j in range(len(allMat[0])):
            if allMat[i][j] == 0:
                deltas[i][j] = u[i] + v[j] - cstMat[i][j]
    return deltas

def isOptimal(deltas):
    for row in deltas:
        for val in row:
            if val is not None and val > 0:
                return False
    return True

def newAlloc(allMat, deltas, cstMat):
    """Find entering variable, path, and update allocation."""
    max_pos_val = 0
    start_pos = (-1, -1)
    for i in range(len(deltas)):
        for j in range(len(deltas[0])):
            if deltas[i][j] is not None and deltas[i][j] > max_pos_val:
                max_pos_val = deltas[i][j]
                start_pos = (i, j)

    path = find_path(allMat, start_pos)
    
    if not path or len(path) < 2:
        return allMat, sum(1 for row in allMat for cell in row if cell > 0)
    
    val = float('inf')
    for t in range(1, len(path), 2):
        r, c = path[t]
        if allMat[r][c] > 0:
            val = min(val, allMat[r][c])
    
    if val == float('inf'):
        val = 1
    
    # Update allocations along the path
    for i in range(len(path)):
        r, c = path[i]
        if i % 2 == 0: 
            allMat[r][c] += val
        else: 
            allMat[r][c] -= val
    
    allMat[start_pos[0]][start_pos[1]] = allMat[start_pos[0]][start_pos[1]] if allMat[start_pos[0]][start_pos[1]] > 0 else 0
    
    numAlloc = sum(1 for row in allMat for cell in row if cell > 0)
    return allMat, numAlloc

def isIndependentAllocation(allocMatrix):
    """Check if allocation is independent using elimination method."""
    elimRows = [0] * len(allocMatrix)
    elimCols = [0] * len(allocMatrix[0])
    
    while True:
        flag = 0
        for i in range(len(allocMatrix)):
            if elimRows[i] == 0:
                allocs = len([allocMatrix[i][j] for j in range(len(allocMatrix[0])) 
                             if elimCols[j] == 0 and allocMatrix[i][j] > 0])
                if allocs < 2:
                    elimRows[i] = 1
                    flag = 1
        
        for j in range(len(allocMatrix[0])):
            if elimCols[j] == 0:
                allocs = len([allocMatrix[i][j] for i in range(len(allocMatrix)) 
                             if elimRows[i] == 0 and allocMatrix[i][j] > 0])
                if allocs < 2:
                    elimCols[j] = 1
                    flag = 1
        
        if flag == 0:
            if 0 not in elimRows and 0 not in elimCols:
                return True, elimRows, elimCols
            else:
                return False, elimRows, elimCols

def find_path(allMat, start_pos):
    """Find the path using elimination method."""
    allMat[start_pos[0]][start_pos[1]] = float('inf')
    
    _, elimRows, elimCols = isIndependentAllocation(allMat)
    rowinds = [i for i in range(len(elimRows)) if elimRows[i] == 0]
    colinds = [i for i in range(len(elimCols)) if elimCols[i] == 0]
    
    path = [list(start_pos)]
    indices = [[x, y] for x in rowinds for y in colinds if allMat[x][y] > 0 and allMat[x][y] != float('inf')]
    
    if path[0] in indices:
        indices.remove(path[0])
    
    n = len(indices) + 1
    while len(path) != n:
        dist = float('inf')
        inds = []
        for i in range(len(indices)):
            d = abs(path[-1][0] - indices[i][0]) + abs(path[-1][1] - indices[i][1])
            if d < dist:
                dist = d
                inds = [indices[i]]
        path.append(inds[0])
        indices.remove(path[-1])
    
    allMat[start_pos[0]][start_pos[1]] = 0
    
    return path

st.set_page_config(layout="wide")
st.title("Transportation Problem Solver (MODI Method)")
st.markdown(
        """
        <hr style="margin-top:50px;margin-bottom:10px;">
        <div style='text-align: center; color: grey; font-size: 14px;'>
            © Made by <b>Musaib Bin Bashir</b> | 2025
        </div>
        """,
        unsafe_allow_html=True
    )

def format_df(df, is_cost_matrix=False):
    """Formats a DataFrame for display."""
    df_display = df.copy()
    if is_cost_matrix:
        df_display.index = [f"Source {i+1}" for i in df_display.index[:-1]] + ["Demand"]
        df_display.columns = [f"Dest {i+1}" for i in df_display.columns[:-1]] + ["Supply"]
    else:
        df_display.index = [f"S{i+1}" for i in df_display.index]
        df_display.columns = [f"D{i+1}" for i in df_display.columns]
    return df_display.astype(str)

st.sidebar.header("Problem Setup")
num_sources = st.sidebar.number_input("Enter the number of sources:", min_value=1, value=3)
num_dests = st.sidebar.number_input("Enter the number of destinations:", min_value=1, value=4)
allocation_method = st.sidebar.selectbox(
    "Choose Initial Allocation Method:",
    ["NWCM (North-West Corner)", "LCM (Least Cost)", "VAM (Vogel's Approximation)"],
    index=0
)

st.subheader("1. Enter Costs, Supply, and Demand")

cost_supply_df = pd.DataFrame(np.zeros((num_sources, num_dests + 1)),
                              index=[f"Source {i+1}" for i in range(num_sources)],
                              columns=[f"Cost to Dest {j+1}" for j in range(num_dests)] + ["Supply"])

demand_df = pd.DataFrame([np.zeros(num_dests)], 
                         columns=[f"Demand at Dest {j+1}" for j in range(num_dests)],
                         index=["Demand"])

st.write("Costs and Supply")
edited_costs_df = st.data_editor(cost_supply_df, num_rows="dynamic")

st.write("Demand")
edited_demand_df = st.data_editor(demand_df)
    
if st.button("Calculate Optimal Solution", type="primary"):
    try:
        costs = edited_costs_df.iloc[:, :-1].values.tolist()
        supply = edited_costs_df.iloc[:, -1].values.tolist()
        demand = edited_demand_df.iloc[0, :].values.tolist()

        costMatrix = []
        for i in range(len(costs)):
            costMatrix.append(costs[i] + [supply[i]])
        costMatrix.append(demand + [0])
        
        st.subheader("2. Initial Problem")
        initial_df = pd.DataFrame(costMatrix)
        st.dataframe(format_df(initial_df, is_cost_matrix=True))

        # Balancing
        is_bal, total_supply, total_demand = isBalanced(costMatrix)
        st.markdown(f"**Total Supply:** `{total_supply}` | **Total Demand:** `{total_demand}`")
        if not is_bal:
            st.warning("Problem is unbalanced. Balancing...")
            costMatrix = balanceProblem(costMatrix, total_supply, total_demand)
            st.subheader("Balanced Cost Matrix")
            balanced_df = pd.DataFrame(costMatrix)
            st.dataframe(format_df(balanced_df, is_cost_matrix=True))
        else:
            st.success("Problem is balanced.")
            
        #Initial Solution
        initial_cost, num_allocated, alloc_matrix = getTotalCost(costMatrix, method=allocation_method.split()[0])
        st.subheader(f"3. Initial Feasible Solution ({allocation_method})")
        alloc_df = pd.DataFrame(alloc_matrix)
        st.dataframe(format_df(alloc_df, is_cost_matrix=True))
        st.info(f"**Initial Feasible Cost:** `{initial_cost}`")

        cstMat = [row[:-1] for row in costMatrix[:-1]]
        allMat = [row[:-1] for row in alloc_matrix[:-1]]
        
        iteration = 1
        st.subheader("4. Finding Optimal Solution (MODI Method)")

        #Optimization Loop
        while True:
            with st.expander(f"**Iteration {iteration}**", expanded=True):
                is_deg = isDegenerate(costMatrix, num_allocated)
                m, n = len(allMat), len(allMat[0])
                st.markdown(f"**Allocations:** `{num_allocated}`. **Required for non-degenerate:** `{m + n - 1}`")

                if is_deg:
                    st.warning("Degenerate solution detected. Resolving...")
                    allMat = removeDeg(allMat, cstMat)
                    num_allocated += 1
                    st.write("Allocation Matrix after resolving degeneracy:")
                    st.dataframe(format_df(pd.DataFrame(allMat)))
                else:
                    st.success("Solution is non-degenerate.")
                
                u, v = findUV(cstMat, allMat)
                deltas = findDeltas(cstMat, allMat, u, v)

                st.write("**Current Allocation**")
                alloc_uv_df = pd.DataFrame(allMat).round(2)
                alloc_uv_df.columns = [f"v{j+1}={v[j]}" for j in range(len(v))]
                alloc_uv_df.index = [f"u{i+1}={u[i]}" for i in range(len(u))]
                st.dataframe(alloc_uv_df)

                st.write("**Reduced Costs (Δij = ui + vj - Cij)** for non-basic variables (unallocated cells):")
                deltas_df = pd.DataFrame(deltas).round(2).replace(np.nan, "")
                st.dataframe(format_df(deltas_df))
                
                if isOptimal(deltas):
                    st.success("**Optimal Solution Found!**")
                    break
                else:
                    max_pos_val = 0
                    entering_var = (-1, -1)
                    for i in range(len(deltas)):
                        for j in range(len(deltas[0])):
                            if deltas[i][j] is not None and deltas[i][j] > max_pos_val:
                                max_pos_val = deltas[i][j]
                                entering_var = (i, j)
                    
                    st.write(f"**Entering Variable:** Cell {entering_var} with reduced cost {max_pos_val:.2f}")
                    
                    path = find_path(allMat, entering_var)
                    st.write(f"**Path found:** {path}")
                    
                    if not path or len(path) < 2:
                        st.error(f"Cannot find valid loop!")
                        break
                    
                    st.info("Solution is not optimal. Re-allocating for the next iteration...")
                    allMat, num_allocated = newAlloc(allMat, deltas, cstMat)
                    iteration += 1
        
        #Final Result
        st.subheader("5. Final Optimal Solution")
        final_alloc_df = pd.DataFrame(allMat)
        st.dataframe(format_df(final_alloc_df))

        optimal_cost = 0
        for i in range(len(cstMat)):
            for j in range(len(cstMat[0])):
                alloc_val = allMat[i][j] if allMat[i][j] > 1e-6 else 0
                optimal_cost += cstMat[i][j] * alloc_val

        st.markdown(f"## **Optimal Cost = `{optimal_cost:.2f}`**")

    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        st.error("Please check your inputs. All values must be numeric.")

st.markdown(
        """
        <hr style="margin-top:50px;margin-bottom:10px;">
        <div style='text-align: center; color: grey; font-size: 14px;'>
            © Made by <b>Musaib Bin Bashir</b> | 2025
        </div>
        """,
        unsafe_allow_html=True
    )

    

