from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
parkinsons_telemonitoring = fetch_ucirepo(id=189) 
  
# data (as pandas dataframes) 
X = parkinsons_telemonitoring.data.features 
y = parkinsons_telemonitoring.data.targets 
  
# metadata 
print(parkinsons_telemonitoring.metadata) 
  
# variable information 
print(parkinsons_telemonitoring.variables) 

dataset = fetch_ucirepo(id=189)

X = dataset.data.features
y = dataset.data.targets

print("Chargement OK")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Aperçu X:")
print(X.head())
print("Aperçu y:")
print(y.head())

assert not X.empty, "X est vide"
assert not y.empty, "y est vide"