import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("../data/train.csv")

def test():
    print("Test")

def survive_ratio():
    survived = len(train.loc[train.Survived == 1])
    total = len(train)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Survived', 'Died'
    sizes = [survived, total-survived]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1.savefig("../img/survive_ratio.png")
    plt.show()

def survive_ratio_by_gender():
    women = train.loc[train.Sex == 'female']["Survived"]
    men = train.loc[train.Sex == 'male']["Survived"]

    # creating the dataset
    labels = ['Female', 'Men']
    ratios = [sum(women)/len(women), sum(men)/len(men)]
    ratios_fmt = ["{:.2%}".format(val) for val in ratios]
    
    fig = plt.figure(figsize = (6, 4))
    plt.bar(labels, ratios, color ='lightblue', width = 0.4)
    for i in range(len(ratios)):
        plt.text(i, ratios[i], ratios_fmt[i], ha = 'center')

    # plt.xlabel("Ratio surviver by gender")
    plt.ylabel("Percentage")
    plt.title("Surviving Ratio by gender")
    plt.savefig("../img/surviving_rate_by_gender.png", dpi=100)

def age_distribution():
    bins= [0,2,4,13,20,110]
    labels = ['Infant','Toddler','Kid','Teen','Adult']

    age_groups = pd.cut(train['Age'], bins=bins, labels=labels, right=False, retbins=True)

    data = {l : 0 for l in labels}
    data["Unknown"] = age_groups[0].size - age_groups[0].count()
    for _, value in age_groups[0].dropna().items():
        data[value] += 1


    # creating the dataset
    labels = list(data.keys())
    values = list(data.values())
    
    plt.figure(figsize = (6, 4))
    plt.bar(labels, values, color ='lightblue', width = 0.4)
    for i in range(len(values)):
        plt.text(i, values[i], values[i], ha = 'center')

    # plt.xlabel("Ratio surviver by gender")
    plt.ylabel("Number")
    plt.title("Age distribution of training data")
    plt.savefig("../img/age_distribution.png", dpi=100)

def age_survival_by_group():
    bins= [0,2,4,13,20,110]
    labels = ['Infant','Toddler','Kid','Teen','Adult']

    age_groups = pd.cut(train['Age'], bins=bins, labels=labels, right=False, retbins=True)[0]
    train['AgeGroup'] = age_groups
    train['AgeGroup'].replace(np.nan, 'Unknown', inplace=True)

    groupByAge = train.groupby('AgeGroup')
    survival = groupByAge['Survived'].value_counts()
    survivedData = {}
    diedData = {}
    for l, v in survival.items():
        if l[1] == 0:
            diedData[l[0]] = v
        else:
            survivedData[l[0]] = v

    df = pd.DataFrame({
        'Age Group' : list(survivedData.keys()),
        'Survived' : list(survivedData.values()),
        'Died' : list(diedData.values())
    })
    fig = df.plot(x="Age Group", y=["Survived", "Died"], color=["g","r"], kind="bar", figsize=(14, 8)).get_figure()
    fig.savefig("../img/survive_vs_dead_by_age_greoup.png")