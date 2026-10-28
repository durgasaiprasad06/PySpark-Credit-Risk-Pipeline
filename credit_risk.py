import warnings
warnings.filterwarnings("ignore")  # Suppresses standard Python warnings

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer, StringIndexer,
    OneHotEncoder, VectorAssembler
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# Start Spark
spark = SparkSession.builder \
    .appName("Credit Risk Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Suppresses all PySpark Java/DAGScheduler warnings (Forces it to only show Errors)
spark.sparkContext.setLogLevel("ERROR")

# Load Dataset
df = spark.read.csv(
    "credit_risk_dataset.csv",
    header=True,
    inferSchema=True
)
df = df.dropDuplicates()
df = df.withColumn(
    "loan_status",
    col("loan_status").cast("double")
)
df = df.dropna(subset=["loan_status"])
df = df.fillna("Unknown", subset=[
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
])

print("Dataset Schema:")
df.printSchema()
print("Total Records:", df.count())

# Chart 1: Loan Default Distribution
status = df.groupBy("loan_status").count().toPandas()
status["loan_status"] = status["loan_status"].map(
    {0.0: "No Default", 1.0: "Default"}
)
plt.figure(figsize=(7, 5))
plt.bar(status["loan_status"], status["count"])
plt.title("Loan Default Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("outputs/chart1_loan_status.png")
plt.close()

# Chart 2: Loan Purpose
intent = df.groupBy("loan_intent").count().toPandas()
plt.figure(figsize=(8, 5))
plt.bar(intent["loan_intent"], intent["count"])
plt.title("Loan Applications by Purpose")
plt.xlabel("Loan Purpose")
plt.ylabel("Applications")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/chart2_loan_purpose.png")
plt.close()

# Chart 3: Loan Grade
grade = df.groupBy("loan_grade").count().toPandas()
grade = grade.sort_values("loan_grade")
plt.figure(figsize=(7, 5))
plt.bar(grade["loan_grade"], grade["count"])
plt.title("Loan Grade Distribution")
plt.xlabel("Loan Grade")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("outputs/chart3_loan_grade.png")
plt.close()

# Chart 4: Applicant Income
income = df.select("person_income").dropna().toPandas()
plt.figure(figsize=(7, 5))
plt.hist(income["person_income"], bins=30)
plt.title("Applicant Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/chart4_income.png")
plt.close()

# Feature Selection
numeric_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[c + "_imputed" for c in numeric_cols],
    strategy="median"
)

indexer = StringIndexer(
    inputCols=categorical_cols,
    outputCols=[c + "_index" for c in categorical_cols],
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=[c + "_index" for c in categorical_cols],
    outputCols=[c + "_encoded" for c in categorical_cols],
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=[c + "_imputed" for c in numeric_cols] +
              [c + "_encoded" for c in categorical_cols],
    outputCol="features"
)

# Train-Test Split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Classification Models
lr = LogisticRegression(
    featuresCol="features",
    labelCol="loan_status",
    maxIter=50
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="loan_status",
    numTrees=100,
    maxDepth=8,
    seed=42
)

# Train Models
lr_model = Pipeline(
    stages=[imputer, indexer, encoder, assembler, lr]
).fit(train)

rf_model = Pipeline(
    stages=[imputer, indexer, encoder, assembler, rf]
).fit(train)

lr_predictions = lr_model.transform(test)
rf_predictions = rf_model.transform(test)

# Evaluation
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="loan_status",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="loan_status",
    predictionCol="prediction",
    metricName="f1"
)

auc_eval = BinaryClassificationEvaluator(
    labelCol="loan_status",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

results = []

for name, predictions in [
    ("Logistic Regression", lr_predictions),
    ("Random Forest", rf_predictions)
]:
    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    auc = auc_eval.evaluate(predictions)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

# Chart 5: Model Comparison
results_df = pd.DataFrame(results).set_index("Model")
results_df.plot(kind="bar", figsize=(9, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/chart5_model_comparison.png")
plt.close()

# Save Predictions
rf_predictions.select(
    "loan_status",
    "prediction",
    col("probability").cast("string").alias("probability")
).write.mode("overwrite").option(
    "header", True
).csv("outputs/loan_predictions")

# Save Models
lr_model.write().overwrite().save("outputs/logistic_model")
rf_model.write().overwrite().save("outputs/random_forest_model")

spark.stop()