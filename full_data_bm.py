
#!/usr/bin/env python

###################################
# IMPORTS
###################################
import re
import ast
import time
import numpy as np
import pandas as pd
from pyspark.sql import Row

###################################
# SETUP SPARK
###################################
# start Spark Session
from pyspark.sql import SparkSession
app_name = "final_project"
# master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .config('spark.executor.memory', '10g')\
        .getOrCreate()
# from pyspark import SparkContext
# SparkContext.setSystemProperty('spark.executor.memory', '15g')
sc = spark.sparkContext

###################################
# TRAIN/TEST Split
###################################
from pyspark.sql.functions import col, when, log, udf, log1p
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.sql.types import FloatType
def log_loss_from_prediction(predictions):
    # predictions are what returns from model.transform
    # the data frame should have a column named probability, which is a tuple:
    # we need to extract the second item of the tuple and calculate log loss with it
    epsilon = 1e-16
    split1_udf = udf(lambda value: value[1].item(), FloatType())
    predictions = predictions.select('*', split1_udf('probability').\
                                     alias('prob'))
    loss = predictions.select("*", 
                           when(predictions.label == 1, 0. - log(predictions.prob + epsilon)).\
                           otherwise(0. - log(1. - predictions.prob + epsilon)).\
                           alias('log_loss')).\
                agg({'log_loss': 'avg'}).\
                take(1)
    return loss

def f1_score(predictions):
    # predictions are what returns from model.transform
    # an exmaple of use:
    # predictions = lrModel.transform(test)
    # x = f1_score(predictions)
    TN = predictions.filter('prediction = 0 AND label = prediction').count()
    TP = predictions.filter('prediction = 1 AND label = prediction').count()
    FN = predictions.filter('prediction = 0 AND label <> prediction').count()
    FP = predictions.filter('prediction = 1 AND label <> prediction').count()
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F =  2 * (precision*recall) / (precision + recall)
    return F

def impute_blank(x):
        if x in ['b03', 'b03', 'b12', 'b16', 'b21', 'b24']:
            impute = 'empty_1'
        elif x in ['b19', 'b20', 'b25', 'b26']:
            impute = 'empty_2'
        else:
            impute = 'empty_0'

        return when(col(x) != "", col(x)).otherwise(impute)

def impute_1pc(x, larger_than_1pc):
    return when(col(x).isin(list(larger_than_1pc)[0]), col(x)).otherwise('less_than_1pc')

def log_transformation(x):
    return when(col(x) < 0, col(x)).otherwise(log1p(col(x)))

parquet_df = spark.read.parquet("gs://261_bucket_zengm71/full_data/train.parquet")
# parquet_df =  parquet_df.sample(fraction=100000/(34095179 + 11745438), seed=8888).cache()

sample_df = parquet_df.sample(fraction=100000/(34095179 + 11745438), seed=8888).cache()
print(sample_df.count())
sample_df_pd = sample_df.toPandas()

categorical_features = [t[0] for t in sample_df.dtypes if t[1] == 'string']
features_summary = pd.DataFrame(columns=['name', '# unique', '# empty', 
                                         '# count = 1', '# count < 10', '# count < 100', 
                                         '# count < 1000'])
for c in categorical_features:
    # number of categories
    nc = len(sample_df_pd.loc[:, c].unique())
    # number of empty strings
    ne = sum(sample_df_pd.loc[:, c] == '')
    # number of categories with only 1 counts
    n1 = sum(sample_df_pd.loc[:, c].value_counts() == 1)
    # number of categories with less than 10 occurances
    n10 = sum(sample_df_pd.loc[:, c].value_counts() < 10)
    # number of categories with less than 100 occurances
    n100 = sum(sample_df_pd.loc[:, c].value_counts() < 100)
    # number of categories with less than 1000 occurances, which is about 1%
    n1000 = sum(sample_df_pd.loc[:, c].value_counts() < 1000)

    features_summary.loc[-1] = [c, nc, ne, n1, n10, n100, n1000]
    features_summary.index = features_summary.index + 1

categorical_features_select = []
features_summary = pd.DataFrame(columns=['name', '# unique before', '# unique after', 
                                         'larger_than_1pc'])
for c in categorical_features:
    # number of categories
    nc_bf = len(sample_df_pd.loc[:, c].unique())

    if c in ['b03', 'b03', 'b12', 'b16', 'b21', 'b24']:
        sample_df_pd.loc[sample_df_pd.loc[:, c] == '', c] = 'empty_1'
    elif c in ['b19', 'b20', 'b25', 'b26']:
        sample_df_pd.loc[sample_df_pd.loc[:, c] == '', c] = 'empty_2'
    else:
        sample_df_pd.loc[sample_df_pd.loc[:, c] == '', c] = 'empty_0'

    less_than_1000 = list(sample_df_pd.loc[:, c].value_counts()[sample_df_pd.loc[:, c].value_counts() < 1000].index)
    sample_df_pd.loc[sample_df_pd.loc[:, c].isin(less_than_1000), c] = 'less_than_1pc'
    larger_than_1pc = [x for x in sample_df_pd.loc[:, c].unique() if x not in ['less_than_1pc']]          
    nc_af = len(sample_df_pd.loc[:, c].unique())

    features_summary.loc[-1] = [c, nc_bf, nc_af, larger_than_1pc]
    features_summary.index = features_summary.index + 1

categorical_features = [t[0] for t in parquet_df.dtypes if t[1] == 'string']
numeric_features = [t[0] for t in parquet_df.dtypes if t[1] == 'double']

for c in categorical_features:
    # every categorical feature:
    # 1) replace empty string with name_na
    # 2) replace categories with less than 1pc with a string
    larger_than_1pc = features_summary.larger_than_1pc[features_summary.name == c]
    parquet_df = parquet_df.withColumn(c, impute_blank(c))\
                           .withColumn(c, impute_1pc(c, larger_than_1pc))

train_bm, test_bm = parquet_df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train_bm.count()))
print("Test Dataset Count: " + str(test_bm.count()))


###################################
# Breiman Transformation
###################################
print('starting Breiman Transformation')
from pyspark.sql.functions import broadcast

for c in categorical_features:
    print(c)
    means = train_bm.groupBy(c).agg({'label':'mean'})
    means = means.withColumnRenamed('avg(label)', c+'_bm')
    means = means.withColumnRenamed(c, 'r')

    train_bm = train_bm.withColumnRenamed(c, 'l')
    train_bm.repartition('l')
    train_bm = train_bm.join(broadcast(means), train_bm.l == means.r, how = 'left').drop('l').drop('r')

    test_bm = test_bm.withColumnRenamed(c, 'l')
    test_bm.repartition('l')
    test_bm = test_bm.join(broadcast(means), test_bm.l == means.r, how = 'left').drop('l').drop('r')
    
    if c in ['b05', 'b10', 'b15', 'b20', 'b26']:
        print('repartition')
        train_bm.write.parquet("full_data/temp/trainbm.parquet" + c)
        test_bm.write.parquet("full_data/temp/testbm.parquet"+ c)
        train_bm = spark.read.parquet("full_data/temp/trainbm.parquet"+ c)
        test_bm = spark.read.parquet("full_data/temp/testbm.parquet"+ c)
        
test_bm = test_bm.na.fill(0)
print('finished Breiman Transformation')
print(train_bm.describe().toPandas().transpose())

print('starting pipeline')
for n in numeric_features + [b + '_bm' for b in categorical_features]:
    # log1p on all numerical features
    # same for train and test
    train_bm = train_bm.withColumn(n, log_transformation(n))
    test_bm = test_bm.withColumn(n, log_transformation(n))
stages = []
assemblerInputs =  [c + "_bm" for c in categorical_features] +numeric_features  #
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
scaler = StandardScaler(inputCol='features', outputCol='selected_features',
                        withStd=True, withMean=True)
stages += [scaler]
stages
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(train_bm)
train = pipelineModel.transform(train_bm).select(['label', 'selected_features'])
train.printSchema()
test = pipelineModel.transform(test_bm).select(['label', 'selected_features'])
test.printSchema()
from pyspark.sql.functions import isnan, when, count, col
train_bm.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in train_bm.columns]).collect()
test_bm.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in test_bm.columns]).collect()



print("====== Training Dataset Count: " + str(train.count()))
print("====== Training Dataset Features Count " + str(len(train.select('selected_features').take(1)[0][0])))
print("====== Test Dataset Count: " + str(test.count()))
train.write.parquet("gs://261_bucket_zengm71/full_data/train.pipe.parquet.bm")
test.write.parquet("gs://261_bucket_zengm71/full_data/test.pipe.parquet.bm")


###################################
# LOGISTIC REGRESSION
###################################
print("====== Logistic Regression ========================================")
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lr = LogisticRegression(featuresCol = 'selected_features', labelCol = 'label', 
                        maxIter=100, regParam=0.001, elasticNetParam=0.5)
lrModel = lr.fit(train)
trainingSummary = lrModel.summary

# Evaluate on Train
predictions = lrModel.transform(train)
evaluator = BinaryClassificationEvaluator()
f1 = f1_score(predictions)
print('====== Train Area Under ROC', evaluator.evaluate(predictions))
print('====== Train Log Loss: ', log_loss_from_prediction(predictions))
print('====== Train F1 Score: ', f1)

# Make Predictions
predictions = lrModel.transform(test)
evaluator = BinaryClassificationEvaluator()
f1 = f1_score(predictions)
print('====== Test Area Under ROC', evaluator.evaluate(predictions))
print('====== Test Log Loss: ', log_loss_from_prediction(predictions))
print('====== Test F1 Score: ', f1)


# # ###################################
# # # GBT
# # ###################################
# print("====== Gradient Boosted Tress ========================================")
# from pyspark.ml.classification import GBTClassifier
# gb = GBTClassifier(featuresCol = 'selected_features', labelCol = 'label', 
#                    maxIter = 100, seed = 8888)
# gbModel = gb.fit(train)

# # Evaluate on Train
# predictions = gbModel.transform(train)
# evaluator = BinaryClassificationEvaluator()
# print('====== Train Area Under ROC', evaluator.evaluate(predictions))
# print('====== Train Log Loss: ', log_loss_from_prediction(predictions))

# # Make Predictions
# predictions = gbModel.transform(test)
# evaluator = BinaryClassificationEvaluator()
# print('====== Test Area Under ROC', evaluator.evaluate(predictions))
# print('====== Test Log Loss: ', log_loss_from_prediction(predictions))

spark.stop()
