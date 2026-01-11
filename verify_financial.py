
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.financial.engine import FinancialCoordinator
    print("SUCCESS: Import FinancialCoordinator")
    
    coord = FinancialCoordinator()
    print("SUCCESS: Initialize Coordinator")
    
    # Optional: Mock run if we don't want to hit APIs
    # result = coord.analyze_company("Tesla", window=5) 
    # print(f"Analysis Result Keys: {result.keys()}")
    
except Exception as e:
    print(f"FAILURE: {e}")
