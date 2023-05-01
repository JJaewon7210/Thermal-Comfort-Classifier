import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import openpyxl
# import file
NAME = 'Label_two_230419_104145'  # Classifier_two_230418_212316
df = pd.read_excel(f'Classfiers/{NAME}.xlsx',
                   index_col=0, sheet_name='Sheet1')
df_headpose = pd.read_csv(
    'D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv', index_col=0)

# set index
df = df.set_index('id_test')
df_headpose = df_headpose.set_index('ID')

# merge
merged_df = pd.merge(df, df_headpose, left_index=True, right_index=True, how='outer')
print(len(merged_df))
df = merged_df.reset_index(drop=True)
df = df.drop(columns = [f'x{i}' for i in range(73)])
df = df.drop(columns = [f'y{i}' for i in range(73)])
df = df.drop(columns = [f'pred_x{i}' for i in range(68)])
df = df.drop(columns = [f'pred_y{i}' for i in range(68)])
df.to_excel(f'Classfiers/{NAME}_2.xlsx')

df['Correct'] = df['MLPClassifier'] == df['y_test']
df['pred_pose0_abs'] = df['pred_pose0'].apply(lambda x: abs(x))
df['pred_pose1_abs'] = df['pred_pose1'].apply(lambda x: abs(x))
df['pred_pose2_abs'] = df['pred_pose2'].apply(lambda x: abs(x))
df['pred_pose1_norm'] = np.linalg.norm(
    df[['pred_pose0', 'pred_pose2', 'pred_pose1']], axis=1)
df['pred_pose2_norm'] = np.linalg.norm(df[['pred_pose1', 'pred_pose0']], axis=1)

# create bins for predpose ranges
bins = [-float('inf'), -30, -15, 0, 15, 30, float('inf')]

# create labels for each range
labels = ['<-30', '-30 to -15', '-15 to 0', '0 to 15', '15 to 30', '>30']

# create a new column to store the range label for each predpose value
df['range0'] = pd.cut(df['pred_pose0'], bins=bins, labels=labels)
df['range1'] = pd.cut(df['pred_pose1'], bins=bins, labels=labels)
df['range2'] = pd.cut(df['pred_pose2'], bins=bins, labels=labels)
df['range0_abs'] = pd.cut(df['pred_pose0_abs'], bins=bins, labels=labels)
df['range1_abs'] = pd.cut(df['pred_pose1_abs'], bins=bins, labels=labels)
df['range2_abs'] = pd.cut(df['pred_pose2_abs'], bins=bins, labels=labels)
df['range1_norm'] = pd.cut(df['pred_pose1_norm'], bins=bins, labels=labels)
df['range2_norm'] = pd.cut(df['pred_pose2_norm'], bins=bins, labels=labels)

# count the number of rows for each range
counts_total0 = df['range0'].value_counts()
counts_total1 = df['range1'].value_counts()
counts_total2 = df['range2'].value_counts()

counts_total0_abs = df['range0_abs'].value_counts()
counts_total1_abs = df['range1_abs'].value_counts()
counts_total2_abs = df['range2_abs'].value_counts()
counts_total1_norm = df['range1_norm'].value_counts()
counts_total2_norm = df['range2_norm'].value_counts()
# count the number of rows where Correct is True for each range
counts_true0 = df.loc[df['Correct'] == True, 'range0'].value_counts()
counts_true1 = df.loc[df['Correct'] == True, 'range1'].value_counts()
counts_true2 = df.loc[df['Correct'] == True, 'range2'].value_counts()

counts_true0_abs = df.loc[df['Correct'] == True, 'range0_abs'].value_counts()
counts_true1_abs = df.loc[df['Correct'] == True, 'range1_abs'].value_counts()
counts_true2_abs = df.loc[df['Correct'] == True, 'range2_abs'].value_counts()
counts_true1_norm = df.loc[df['Correct'] == True, 'range1_norm'].value_counts()
counts_true2_norm = df.loc[df['Correct'] == True, 'range2_norm'].value_counts()

# concatenate the counts and accuracy results for each column
counts = pd.concat([counts_true0, counts_total0, counts_true0_abs, counts_total0_abs,
                    counts_true1, counts_total1, counts_true1_abs, counts_total1_abs,
                    counts_true2, counts_total2, counts_true2_abs, counts_total2_abs,
                    counts_true1_norm, counts_total1_norm, counts_true2_norm, counts_total2_norm],
                   axis=1,
                   keys=['true0', 'total0', 'true0_abs', 'total0_abs',
                         'true1', 'total1', 'true1_abs', 'total1_abs',
                         'true2', 'total2', 'true2_abs', 'total2_abs',
                         'true1_norm', 'total1_norm', 'true2_norm', 'total2_norm'])

counts['accuracy0'] = counts['true0'] / counts['total0']
counts['accuracy1'] = counts['true1'] / counts['total1']
counts['accuracy2'] = counts['true2'] / counts['total2']
counts['accuracy0_abs'] = counts['true0_abs'] / counts['total0_abs']
counts['accuracy1_abs'] = counts['true1_abs'] / counts['total1_abs']
counts['accuracy2_abs'] = counts['true2_abs'] / counts['total2_abs']
counts['accuracy1_norm'] = counts['true1_norm'] / counts['total1_norm']
counts['accuracy2_norm'] = counts['true2_norm'] / counts['total2_norm']

counts = counts.reindex(labels)
print(counts)
# create a Pandas Excel writer using the existing file
with pd.ExcelWriter(f'Classfiers/{NAME}_2.xlsx', mode='a') as writer:
    # write the counts to a new sheet
    counts.to_excel(writer, sheet_name='pose_analyze')
