#pd
# .value_counts()
# inplace=True
# .get_dummies -  creating a categorical variables (from C,s,M to 0/1 on C S, M)
# final_train["Age"][final_train.Survived == 1] z dataframu final_train wybierasz te "age w których Survived==1


# np
# np.where(condition  (+=OR),1,0)

# sns
# sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)

# For numerical visualisations:
# plt.figure(figsize=(15, 8))
# ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", shade=True)
# sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", shade=True)
# plt.legend(['Survived', 'Died'])
# plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# ax.set(xlabel='Fare')
# plt.xlim(-20, 200)
# plt.show()

# For categorical visualisations:
# sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")
# plt.show()