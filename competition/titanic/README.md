# Titanic - Machine Learning from Disaster

## Overview
The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. In this challenge, we ask you to build a predictive model that answers the question: **“what sorts of people were more likely to survive?”** using passenger data (ie name, age, gender, socio-economic class, etc).

## Objective
* **Task:** Binary Classification
* **Target Variable:** `Survived` (0 = No, 1 = Yes)
* **Evaluation Metric:** Categorization Accuracy (Percentage of passengers correctly predicted)

## Dataset Description
The dataset is split into two groups:
* `train.csv` (891 rows): Contains details of a subset of the passengers on board and crucially, whether they survived or not.
* `test.csv` (418 rows): Contains similar passenger details but the ground truth `Survived` is withheld. Your task is to predict these labels.
* `gender_submission.csv`: An example submission file predicting that all female passengers survived and all male passengers died.

## Data Dictionary
| Variable | Definition | Key |
| :--- | :--- | :--- |
| **PassengerId** | Unique identifier for each passenger | 1 to 1309 |
| **Survived** | Survival | 0 = No, 1 = Yes |
| **Pclass** | Ticket class (Socio-economic status) | 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower) |
| **Name** | Full passenger name with title | e.g. "Braund, Mr. Owen Harris" |
| **Sex** | Gender | male, female |
| **Age** | Age in years | Fractional if less than 1; xx.5 if estimated |
| **SibSp** | # of siblings / spouses aboard the Titanic | Integer |
| **Parch** | # of parents / children aboard the Titanic | Integer |
| **Ticket** | Ticket number string | Alphanumeric string |
| **Fare** | Passenger fare paid | Float |
| **Cabin** | Cabin number | String (e.g. C85, B96, etc.) |
| **Embarked** | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

## Submission Format
Your submission file should be in CSV format with exactly 418 entries plus a header row, matching `gender_submission.csv`:
```csv
PassengerId,Survived
892,0
893,1
894,0
...
```
