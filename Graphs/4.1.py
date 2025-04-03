import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/threshold_policy_costs_first_model.csv")  # or your most recent policy cost output

plt.figure(figsize=(10, 6))
plt.plot(df["Threshold"], df["Truck_Cost"], marker='o', label="Truck Rental Cost")
plt.plot(df["Threshold"], df["3PL_Cost"], marker='s', label="3PL Cost")
plt.xlabel("Shipment Threshold (ft³)")
plt.ylabel("Expected Daily Cost ($)")
plt.title("Expected Cost vs Shipment Threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/cost_vs_threshold.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df["Threshold"], df["Expected_Shipments_per_Day"], marker='^', color='green')
plt.xlabel("Shipment Threshold (ft³)")
plt.ylabel("Expected Shipments per Day")
plt.title("Shipment Frequency vs Threshold")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/shipments_per_day_vs_threshold.png")
plt.show()
