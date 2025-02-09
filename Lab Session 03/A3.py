vec1 = X.iloc[0].values
vec2 = X.iloc[1].values
minkowski_distances = []
for r in range(1, 11):
    dist = minkowski(vec1, vec2, r) 
    minkowski_distances.append(dist)
    print(f"Minkowski distance for r={r} is {dist}")
plt.plot(range(1, 11), minkowski_distances, marker="o")
plt.xlabel("Minkowski Order (r)")
plt.ylabel("Distance")
plt.title("Minkowski Distance vs r")
plt.show()
