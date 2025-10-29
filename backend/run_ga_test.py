import pandas as pd
from ga_algorithm import GAAlgorithm

# ======== إعداد الملف والعمود الهدف ========
file = "patients.csv"      # ضع اسم CSV
target = "satisfaction"  # ضع اسم العمود الهدف

# قراءة البيانات
data = pd.read_csv(file)
X = data.drop(columns=[target])
y = data[target]

# اكتشاف الأعمدة النصية
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object" or X[col].dtype.name == "category"]

# ======== تنفيذ خوارزمية GA ========
result = GAAlgorithm.GAOptimize(X, y, cat_features)

# طباعة النتائج
print("أفضل كروموسوم:", result["best_chromosome"])
print("المميزات المختارة:", result["selected_features_indices"])
print("عدد الميزات:", result["num_selected_features"])
print("الدقة:", result["accuracy"])
print("الفيتنس:", result["fitness"])
print("زمن التنفيذ (ثواني):", result["elapsed_time_seconds"])
