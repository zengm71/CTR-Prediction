
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
# PIPE LINE
###################################
if False: 
    # write data to parquet, only needed to be run once
    df = sc.textFile('gs://261_bucket_zengm71/full_data/train.txt').\
            map(lambda l: l.split("\t")).\
            map(lambda p: Row(label=int(p[0]), 
                              c01 = int(p[1] + '0') / 10, c02 = int(p[2] + '0') / 10, c03 = int(p[3] + '0') / 10, c04 = int(p[4] + '0') / 10, c05 = int(p[5] + '0') / 10, c06 = int(p[6] + '0') / 10, 
                              c07 = int(p[7] + '0') / 10, c08 = int(p[8] + '0') / 10, c09 = int(p[9] + '0') / 10, c10 = int(p[10] + '0') / 10, c11 = int(p[11] + '0') / 10, c12 = int(p[12] + '0') / 10, 
                              c13 = int(p[13] + '0') / 10, 
                              b01 = p[14], b02 = p[15], b03 = p[16], b04 = p[17], b05 = p[18], b06 = p[19], b07 = p[20], b08 = p[21], b09 = p[22], b10 = p[23], b11 = p[24], b12 = p[25], b13 = p[26], 
                              b14 = p[27], b15 = p[28], b16 = p[29], b17 = p[30], b18 = p[31], b19 = p[32], b20 = p[33], b21 = p[34], b22 = p[35], b23 = p[36], b24 = p[37], b25 = p[38], b26 = p[39], ))

    # Infer the schema, and register the DataFrame as a table.
    schema_df = spark.createDataFrame(df)
    schema_df.createOrReplaceTempView("df")
    schema_df.write.parquet("gs://261_bucket_zengm71/full_data/train.parquet")

if True:
    # fit the full dataset through the pipeline and write to parquet
    # only need to be run once as well, unless any changes to the pipeline
    ###################################
    # Get the categories
    ###################################
    parquet_df = spark.read.parquet("gs://261_bucket_zengm71/full_data/train.parquet")

    # Parquet files can also be used to create a temporary view and then used in SQL statements.
    parquet_df.createOrReplaceTempView("parquet_df")
    parquet_df.printSchema()

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
    categorical_features_select = categorical_features_select
    sample_df_pd.loc[:, 'b17'].unique()

    numeric_features = [t[0] for t in sample_df.dtypes if t[1] == 'double']

    ###################################
    # Preprocessing
    ###################################
    from pyspark.sql.functions import col, when, log, udf, log1p

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

    from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, StandardScaler, ChiSqSelector
    stages = []
    for categoricalCol in categorical_features_select:
        stringIndexer = StringIndexer(inputCol = categoricalCol, 
                                      outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], 
                                         outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    assemblerInputs = [c + "classVec" for c in categorical_features_select] + numeric_features
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    scaler = StandardScaler(inputCol='features', outputCol='selected_features',
                            withStd=True, withMean=True)
    stages += [scaler]
    # selector = ChiSqSelector(numTopFeatures=50, featuresCol="scaled_features",
    #                          outputCol="selected_features", labelCol="label")
    # stages += [selector]
    stages

    for c in categorical_features_select:
        larger_than_1pc = features_summary.larger_than_1pc[features_summary.name == c]
        parquet_df = parquet_df.withColumn(c, impute_blank(c))\
                               .withColumn(c, impute_1pc(c, larger_than_1pc))

    for n in numeric_features:
        parquet_df = parquet_df.withColumn(n, log_transformation(n))

    print('fitting the pipeline')
    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages = stages)    
    pipelineModelFull = pipeline.fit(sample_df)
    parquet_df_pipe = pipelineModelFull.transform(sample_df)
    selectedCols = ['label', 'selected_features'] #+ cols
    parquet_df_pipe = parquet_df_pipe.select(selectedCols)
    parquet_df_pipe.printSchema()

#     print('writing to train.pipe.parquet')
#     parquet_df_pipe.write.parquet("gs://261_bucket_zengm71/full_data/train.pipe.parquet")

###################################
# TRAIN/TEST Split
###################################
from pyspark.sql.functions import col, when, log, udf, log1p
from pyspark.sql.types import FloatType
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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

    if (TP + FP > 0) and (TP + FN > 0):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F =  2 * (precision*recall) / (precision + recall)
    else:
        F = 0
    return F

def eval(model, train, test, name):
    # this function evaluates the model, and returns a data frame

    print('Model:', name)
    predictions = model.transform(train)
    evaluator = BinaryClassificationEvaluator()
    train_roc = round(evaluator.evaluate(predictions), 6)
#     print('Train Area Under ROC', train_roc)
    ll_train = round(log_loss_from_prediction(predictions)[0][0], 6)
#     print('Train Area Log loss', ll_train)
    F_train = round(f1_score(predictions), 6)
#     print('Train F1-score', F_train)
    
    predictions = model.transform(test)
    test_roc = round(evaluator.evaluate(predictions), 6)
#     print('Test Area Under ROC', test_roc)
    ll_test = round(log_loss_from_prediction(predictions)[0][0], 6)
#     print('Test Area Log loss', ll_test)
    F_test = round(f1_score(predictions), 6)
#     print('Test F1-score', F_test)
    
    eval = pd.DataFrame({'Name': [name, name], 
                         'ROC': [train_roc, test_roc],
                         'LogLoss': [ll_train, ll_test], 
                         'F1-Score': [F_train, F_test]})
    print(eval)
    return(eval)
# parquet_df_pipe = spark.read.parquet("gs://261_bucket_zengm71/full_data/train.pipe.parquet")
train, test = parquet_df_pipe.randomSplit([0.7, 0.3], seed = 2018)
print("====== Training Dataset Count: " + str(train.count()))
print("====== Test Dataset Count: " + str(test.count()))

# ###################################
# # GBT
# ###################################
print("====== Gradient Boosted Tress ========================================")
from pyspark.ml.classification import GBTClassifier
gb = GBTClassifier(featuresCol = 'selected_features', labelCol = 'label', 
                   maxIter = 20, seed = 8888)
gbModel = gb.fit(train)
eval_gbt = eval(gbModel, train, test, 'GBT')

# Create 5-fold CrossValidator
gb = GBTClassifier(featuresCol = 'selected_features', labelCol = 'label', 
                   maxIter = 20, seed = 8888)
paramGrid = ParamGridBuilder()\
    .addGrid(gb.maxDepth,[5, 10])\
    .addGrid(gb.maxBins,[16, 32, 64])\
    .addGrid(gb.maxIter,[50, 100]) \
    .build()

evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator = gb, estimatorParamMaps = paramGrid, 
                    evaluator = evaluator, numFolds=3)
# Run cross validations
cvModel = cv.fit(train)

best_maxDepth = cvModel.bestModel._java_obj.getMaxBins()
best_maxBins = cvModel.bestModel._java_obj.getMaxBins()
best_maxIter = cvModel.bestModel._java_obj.getMaxIter()

print('====== Best maxDepth is', best_maxDepth)
print('====== Best maxBins is', best_maxBins)
print('====== Best maxIter is', best_maxIter)

spark.stop()
