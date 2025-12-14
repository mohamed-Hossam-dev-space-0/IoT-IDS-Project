print("=== IoT IDS Simple Test ===\n")

# Test basic imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")

# Test data generation
print("\nGenerating sample IoT data...")
np.random.seed(42)
data = {
    'packet_size': np.random.normal(500, 150, 100),
    'packet_count': np.random.poisson(5, 100),
    'is_attack': np.random.choice([0, 1], 100, p=[0.8, 0.2])
}
df = pd.DataFrame(data)
print(f"✅ Created {len(df)} samples")
print(f"   Normal: {sum(df['is_attack']==0)}, Attacks: {sum(df['is_attack']==1)}")

# Test AI model
print("\nTesting AI model...")
X = df[['packet_size', 'packet_count']]
y = df['is_attack']

# Simple train/test split
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"✅ AI trained successfully!")
print(f"   Test accuracy: {accuracy*100:.1f}%")

print("\n" + "="*40)
print("SIMPLE TEST COMPLETE - Ready for main project!")
print("="*40)
