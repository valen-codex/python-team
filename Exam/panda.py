import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('student_exam_scores.csv')
sorted_data = data.sort_values(by='exam_score', ascending=False)  
plt.bar(sorted_data['hours_studied'], sorted_data['exam_score'])
# plt.bar(sorted_data['exam_score'])
plt.title('hours studied vs exam score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')    
plt.show()
# print(data.head(5))
#print top 5 of the exam scores
# print(data['exam_score'].head(5))

#sort the data by exam score in ascending order and print the top 5
 
print(sorted_data.tail(5))
