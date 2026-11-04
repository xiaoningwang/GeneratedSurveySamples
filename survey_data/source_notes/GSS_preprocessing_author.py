# -*- coding: utf-8 -*-

###############################################################################
##
## GSS data cleaning (2018)
## Note: Use environment "python3.11"
##
## Created : 11/17/2025
##
###############################################################################


import os
import pandas as pd
import numpy as np


######################### Read in dataset ################################

### Note: I downloaded two versions of the GSS dataset from GSS Data Explorer
# (https://gssdataexplorer.norc.org/MyGSS): one with labels and one with codes,
# and combined them into the final dataset.


# read in dataset with labels
gss = pd.read_csv("READ IN YOUR LABEL DATA FILE HERE")


# read in dataset with codes and rename variables
gss_code = pd.read_csv("READ IN YOUR CODE DATA FILE HERE")

gss_code.columns = [
    col + "_code" if col not in ["year", "id_"] else col
    for col in gss_code.columns
]


# deal with missing codes
missing_codes = [".i:  Inapplicable", 
                 ".y:  Not available in this year",
                 ".x:  Not available in this release",
                 ".n:  No answer",
                 ".m:  DK, NA, IAP",
                 ".p:  Not applicable (I have not faced this decision)/Not imputable",
                 ".d:  Do not Know/Cannot Choose",
                 ".r:  Refused",
                 ".q:  Not imputable",
                 ".u:  Uncodable",
                 ".z:  Variable-specific reserve code",
                 ".s:  Skipped on Web"]

# Replace missing codes with NA
gss = gss.replace(missing_codes, np.nan)


# merge label and code files
gss = pd.merge(gss, gss_code, on=["year", "id_"], how = "left")



######################### Mappings ################################

def gss_clean(df, year, output):
    
    # select year
    df = df[df["year"] == year]
    
    # create empty variable list
    var_lst = []

    # gender
    if not df["sex"].isna().all():
        def gender_mapping(x):
            if x == "MALE":
                return "Male"
            elif x == "FEMALE":
                return "Female"
            else:
                return np.nan
        df["gender"] = df["sex"].apply(gender_mapping)
        var_lst.append("gender")
        print("gender cleaned!")
    else:
        print("no data for gender")
    
    # age
    if not df["age"].isna().all():
        def age_mapping(x):
            if x == "89 or older":
                return "89"
            else:
                return x
        df["age"] = df["age"].apply(age_mapping)
        df["age"] = pd.to_numeric(df["age"], errors="coerce") # make it to numeric
        var_lst.append("age")
        print("age cleaned!")
    else:
        print("no data for age")
    
    # birth year
    df["birth_year"] = np.where(df["age"].notna(), year - df["age"], np.nan)
    var_lst.append("birth_year")

    print("birth_year calculated!")

            
    # race
    if df["race"].notna().any() and df["hispanic"].notna().any(): # 2018
        def race_mapping(x, y):
            if x == "Black":
                return "Black"
            elif y in ['Another Hispanic, Latino, or Spanish origin',
                       'Puerto Rican', 'Mexican, Mexican American, Chicano/a', 
                       'Cuban']:
                return "Hispanic"
            else:
                return "Non-Black, Non-Hispanic"
        df["race"] = df.apply(lambda row: race_mapping(row["race"], row["hispanic"]), axis=1)
        var_lst.append("race")
        print("race cleaned!")
    elif df["race"].notna().any() and df["hispanic"].isna().all(): # used for other years
        var_lst.append("race")
        print("race added!")
    else:
        print("no data for race")
        
        

    # religion
    if df["relig_code"].notna().any():
        def relig_mapping(x):
            if x == 1:
                return "Protestant"
            elif x == 2:
                return "Catholic"
            elif x == 4:
                return "None"
            elif x == 3 or 5 <= x <= 13:
                return "Other"
            else:
                return np.nan
        df["religion"] = df["relig_code"].apply(relig_mapping)
        var_lst.append("religion")
        print("religion cleaned!")
    else:
        print("no data for religion")
            
        
        
    # immigrant status
    if not df["born"].isna().all():
        def migration_mapping(x):
            if x == "YES":
                return "Non-immigrant"
            elif x == "NO":
                return "Immigrant"
            else:
                return np.nan
        df["immigrant_status"] = df["born"].apply(migration_mapping)
        var_lst.append("immigrant_status")
        print("immigrant_status cleaned!")
    else:
        print("no data for immigrant status")
        
        
    # mother's immigrant status:
    if not df["parborn"].isna().all():
        def migration_m_mapping(x):
            if x in ["Both born in the U.S.", "Mother yes, father no", "Mother yes, father don't know"]:
                return "Non-immigrant"
            elif x in ["Neither born in the U.S.", "Mother no, father yes", "Mother no, father don't know"]:
                return "Immigrant"
            else:
                return np.nan
        df["mother_immigrant_status"] = df["parborn"].apply(migration_m_mapping)
        var_lst.append("mother_immigrant_status")
        print("mother_immigrant_status cleaned!")
    else:
        print("no data for mother's immigrant status")

        
    # father immigrant status:
    if not df["parborn"].isna().all():
        def migration_f_mapping(x):
            if x in ["Both born in the U.S.", "Mother no, father yes", "Mother don't know, father yes"]:
                return "Non-immigrant"
            elif x in ["Neither born in the U.S.", "Mother yes, father no", "Mother don't know, father no"]:
                return "Immigrant"
            else:
                return np.nan
        df["father_immigrant_status"] = df["parborn"].apply(migration_f_mapping)
        var_lst.append("father_immigrant_status")
        print("father_immigrant_status cleaned!")
    else:
        print("no data for father's immigrant status")
        
        

    # marital status
    if not df["marital"].isna().all():
        def marital_mapping(x):
            if x in ["Married", "Separated"]:
                return "Married"
            elif x in ["Divorced", "Widowed"]:
                return "Divorced/widowed"
            elif x == "Never married":
                return "Single"
            else:
                return np.nan
        df["marital_status"] = df["marital"].apply(marital_mapping)
        var_lst.append("marital_status")
        print("marital_status cleaned!")
    else:
        print("no data for marital status")
    
    
    
    # child number
    if "childs" in df.columns:
        if not df["childs"].isna().all():
            def nchild_mapping(x):
                if x == "8 or more": 
                    return "8"
                else:
                    return x
            df["child_number"] = df["childs"].apply(nchild_mapping)
            df["child_number"] = pd.to_numeric(df["child_number"], errors="coerce")
            var_lst.append("child_number")
            print("child_number cleaned!")
        else:
            print("no data for child_number")
    

    
    # age at first marriage
    if "agewed" in df.columns and df["agewed"].notna().any():
        df["age_first_marriage"] = pd.to_numeric(df["agewed"], errors="coerce")
        # assign "Never Married" for Single respondents
        df.loc[df["marital_status"] == "Single", "age_first_marriage"] = "Never Married"
    
        var_lst.append("age_first_marriage")
        print("age_first_marriage cleaned!")
    
    else:
        print("no data for age at first marriage")
        
    # alternative estimate of age at first marriage - year of first marriage
    if "marcohrt" in df.columns and df["marcohrt"].notna().any():
        df["marcohrt"] = pd.to_numeric(df["marcohrt"], errors="coerce")
        df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    
        df["age_first_marriage2"] = df["marcohrt"] - df["birth_year"]
    
        # fill missing values in the main variable
        df["age_first_marriage"] = df["age_first_marriage"].fillna(df["age_first_marriage2"])
    

        
    # age at first childbirth
    if "agekdbrn" in df.columns and df["agekdbrn"].notna().any():
        def childbirth_mapping(x):
            if x == "17 and younger": 
                return "17"
            elif x == "45 and older":
                return "45"
            else:
                return x
        df["age_first_childbirth"] = df["agekdbrn"].apply(childbirth_mapping)
        df["age_first_childbirth"] = pd.to_numeric(df["age_first_childbirth"], errors="coerce")
        df.loc[df["child_number"] == 0, "age_first_childbirth"] = "No Child"
    
        var_lst.append("age_first_childbirth")
        print("age_first_childbirth cleaned!")
    else:
        print("no data for age at first childbirth")
    
    
    # cohabit
    if not df["cohabit"].isna().all():
        def cohabit_mapping(x):
            if x == "YES":
                return "Cohabit before marriage"
            elif x == "NO":
                return "Didn't cohabit before marriage"
            else:
                return np.nan
        df["cohabit"] = df["cohabit"].apply(cohabit_mapping)
        var_lst.append("cohabit")
    else:
        print("no data for cohabit")
        

    # cohabit - alternative variable
    if not df["livnowed"].isna().all():
        def cohabit2_mapping(x):
            if x in ["YES, WITH A PREVIOUS PARTNER", 
                     "YES, WITH MY PRESENT PARTNER",
                     "YES, WITH PREVIOUS AND PRESENT PARTNER"]:
                return "Cohabit before marriage"
            elif x == "NO, NEVER":
                return "Didn't cohabit before marriage"
            else:
                return np.nan
        df["cohabit2"] = df["livnowed"].apply(cohabit2_mapping)
        
        # fill cohabit = cohabit2 if cohabit == nan
        df["cohabit"] = df["cohabit"].fillna(df["cohabit2"])
        print("cohabit cleaned!")
 

    # education
    if not df["educ_code"].isna().all():
        def edu_mapping(x):
            if 0 <= x <= 8:
                return "Less than high school"
            elif 9 <= x <= 12:
                return "High school"
            elif x >= 13:
                return "College and above"
            else:
                return np.nan
        df["education"] = df["educ_code"].apply(edu_mapping)   
        var_lst.append("education")
        print("education cleaned!")
    else:
        print("no data for education")
        
    
    # age finish education
    if not df["datesch"].isna().all():
        def finishedu_mapping(x, y):
            if x > 0 and x not in [9998, 9999]:
                # the first two digits indicate the year
                # e.g., "1205" is May 1912
                finishsch_year = x // 100 + 1900
                finishsch_age = finishsch_year - y # y is birthyear
                if 0 <= finishsch_age <= 89:
                    return finishsch_age
                else:
                    return np.nan
                
            else:
                return np.nan

        df["age_finished_education"] = df.apply(lambda row: finishedu_mapping(
            row["datesch_code"], row["birth_year"]), axis=1)
  
        var_lst.append("age_finished_education")
        print("age_finished_education cleaned!")
    else:
        print("no data for age_finished_education")
          
       
    # mother's education
    if not df["maeduc_code"].isna().all():
        def maedu_mapping(x):
            if 0 <= x <= 8:
                return "Less than high school"
            elif 9 <= x <= 12:
                return "High school"
            elif x >= 13:
                return "College and above"
            else:
                return np.nan
        df["mother_education"] = df["maeduc_code"].apply(maedu_mapping)   
        var_lst.append("mother_education")
        print("mother_education cleaned!")
    else:
        print("no data for mother_education")     
         

    # father's education
    if not df["paeduc_code"].isna().all():
        def paedu_mapping(x):
            if 0 <= x <= 8:
                return "Less than high school"
            elif 9 <= x <= 12:
                return "High school"
            elif x >= 13:
                return "College and above"
            else:
                return np.nan
        df["father_education"] = df["paeduc_code"].apply(paedu_mapping)   
        var_lst.append("father_education")
        print("father_education cleaned!")
    else:
        print("no data for father_education")       
        
    # labor force
    if not df["wrkstat_code"].isna().all():
        def labor_mapping(x):
            if 4 <= x <= 8: 
                return "Not in the labor force"
            elif 1 <= x <= 3:
                return "In the labor force"
            else:
                return np.nan
        df["laborforce"] = df["wrkstat_code"].apply(labor_mapping)   
        var_lst.append("laborforce")
        print("laborforce cleaned!")
    else:
        print("no data for laborforce")
        
    
    # occupation
    if not df["occ10_code"].isna().all():
        def occ_mapping(x):
            if 0 <= x <= 3540: 
                return "Management, professional, and related occupations"
            elif 3600 <= x <= 4650:
                return "Service occupations"
            elif 4700 <= x <= 5940:
                return "Sales and office occupations"
            elif 6000 <= x <= 7630:
                return "Natural resources, construction, and maintenance occupations"
            elif 7700 <= x <= 9750:
                return "Production, transportation, and material moving occupations"
            elif 9800 <= x <= 9830:
                return "Military Occupations"
            else:
                return np.nan
        df["occupation"] = df["occ10_code"].apply(occ_mapping)
        
        # add unemployed
        df.loc[df["occupation"].isna() & (df["laborforce"] == "Not in the labor force"),
           "occupation"] = "Unemployed"
        
        var_lst.append("occupation")
        print("occupation cleaned!")
    else:
        print("no data for occupation")
        

    # mother's occupation
    if not df["maocc10_code"].isna().all():
        def maocc_mapping(x):
            if 0 <= x <= 3540: 
                return "Management, professional, and related occupations"
            elif 3600 <= x <= 4650:
                return "Service occupations"
            elif 4700 <= x <= 5940:
                return "Sales and office occupations"
            elif 6000 <= x <= 7630:
                return "Natural resources, construction, and maintenance occupations"
            elif 7700 <= x <= 9750:
                return "Production, transportation, and material moving occupations"
            elif 9800 <= x <= 9830:
                return "Military Occupations"
            else:
                return np.nan
        df["mother_occupation"] = df["maocc10_code"].apply(maocc_mapping)
        
        var_lst.append("mother_occupation")
        print("mother_occupation cleaned!")
    else:
        print("no data for mother_occupation")
    

    # father's occupation
    if not df["paocc10_code"].isna().all():
        def paocc_mapping(x):
            if 0 <= x <= 3540: 
                return "Management, professional, and related occupations"
            elif 3600 <= x <= 4650:
                return "Service occupations"
            elif 4700 <= x <= 5940:
                return "Sales and office occupations"
            elif 6000 <= x <= 7630:
                return "Natural resources, construction, and maintenance occupations"
            elif 7700 <= x <= 9750:
                return "Production, transportation, and material moving occupations"
            elif 9800 <= x <= 9830:
                return "Military Occupations"
            else:
                return np.nan
        df["father_occupation"] = df["paocc10_code"].apply(paocc_mapping)
        
        var_lst.append("father_occupation")
        print("father_occupation cleaned!")
    else:
        print("no data for father_occupation")   
    

    # spouse's occupation
    if not df["spocc10_code"].isna().all():
        def spocc_mapping(x):
            if 0 <= x <= 3540: 
                return "Management, professional, and related occupations"
            elif 3600 <= x <= 4650:
                return "Service occupations"
            elif 4700 <= x <= 5940:
                return "Sales and office occupations"
            elif 6000 <= x <= 7630:
                return "Natural resources, construction, and maintenance occupations"
            elif 7700 <= x <= 9750:
                return "Production, transportation, and material moving occupations"
            elif 9800 <= x <= 9830:
                return "Military Occupations"
            else:
                return np.nan
        df["spouse_occupation"] = df["spocc10_code"].apply(spocc_mapping)
        # assign "Never Married" for Single respondents
        df.loc[df["marital_status"] == "Single", "spouse_occupation"] = "No spouse"
        
        var_lst.append("spouse_occupation")
        print("spouse_occupation cleaned!")
    else:
        print("no data for spouse_occupation")
    
    
    # income
    if not df["rincome"].isna().all():
        def inc_mapping(x):
            if x in ['LT $1000', '$1000 TO 2999',
                     '$3000 TO 3999', '$4000 TO 4999']: 
                return "LT $5000"
            elif x in ['$5000 TO 5999', '$6000 TO 6999', 
                       '$7000 TO 7999', '$8000 TO 9999']:
                return "$5000 TO 9999"
            elif x in ['$15000 - 19999', '$10000 - 14999', '$20000 - 24999',
                   '$25000 OR MORE']:
                return "$10000 OR MORE"
            else:
                return np.nan
        df["income"] = df["rincome"].apply(inc_mapping)  
        
        # assign "Unemployed"
        df.loc[df["occupation"] == "Unemployed", "income"] = "Unemployed"
        
        var_lst.append("income")
        print("income cleaned!")
    else:
        print("no data for income")
    
    
    # wealth
    if not df["wealth_code"].isna().all():
        if not (df["wealth_code"] == -100).all(): # delete missing values
            def wealth_mapping(x):
                if 1 <= x <= 3: 
                    return "LT $40000"
                elif 4 <= x <= 8:
                    return "$40000 TO 500000"
                elif 9 <= x <= 15:
                    return "More than $500000"
                else:
                    return np.nan
            df["wealth"] = df["wealth_code"].apply(wealth_mapping)
            
            var_lst.append("wealth")
            print("wealth cleaned!")
        else:
            print("no data for wealth")
    

    # health status
    if not df["health"].isna().all():
        var_lst.append("health")
        print("health added!")
    else:
        print("no data for health")
    
    
    # depression
    if not df["depress"].isna().all():
        def depress_mapping(x):
            if x == "Yes": 
                return "Had depression"
            elif x == "No":
                return "No depression"
            else:
                return np.nan
        df["depress"] = df["depress"].apply(depress_mapping)  
        var_lst.append("depress")
        print("depress cleaned!")
    else:
        print("no data for depress") 
    
    # gender role
    if not df["fefam_code"].isna().all():
        def genderrole_mapping(x):
            if 1 <= x <= 2: 
                return "Agree"
            elif 3 <= x <= 4:
                return "Disagree"
            else:
                return np.nan
        df["gender_role_attitude"] = df["fefam_code"].apply(genderrole_mapping)  
        var_lst.append("gender_role_attitude")
        print("gender_role_attitude cleaned!")
    else:
        print("no data for gender_role_attitude") 
        

    # happiness
    if not df["happy"].isna().all():
        var_lst.append("happy")
        print("happy added!")
    else:
        print("no data for happy")
        
        
    # Satisfaction with health
    if not df["sathealt"].isna().all():
        df["satisfy_health"] = df["sathealt"]
        var_lst.append("satisfy_health")
        print("satisfy_health added!")
    else:
        print("no data for satisfy_health")
           
        
    # Satisfaction with job
    if not df["satjob"].isna().all():
        df["satisfy_job"] = df["satjob"]
        var_lst.append("satisfy_job")
        print("satisfy_job added!")
    else:
        print("no data for satisfy_job")
        
         
    # isolated
    if not df["lonely2"].isna().all():
        df["isolated"] = df["lonely2"]
        var_lst.append("isolated")
        print("isolated added!")
    else:
        print("no data for isolated")
       
          
    # lonely
    if not df["lonely1"].isna().all():
        df["lonely"] = df["lonely1"]
        var_lst.append("lonely")
        print("lonely added!")
    else:
        print("no data for lonely")
        
        
    # mental health
    if not df["mntlhlth"].isna().all():
        def mental_mapping(x):
            if 0 <= x <= 30: 
                return x
            else:
                return np.nan
        df["mental_health"] = df["mntlhlth_code"].apply(mental_mapping)  
        var_lst.append("mental_health")
        print("mental_health cleaned!")
    else:
        print("no data for mental_health") 
    
    
    # political view
    if not df["polviews_code"].isna().all():
        def poli_mapping(x):
            if 1 <= x <= 3: 
                return "Liberal"
            elif x == 4:
                return "Middle"
            elif 5 <= x <= 7:
                return "Conservative"
            else:
                return np.nan
        df["political_view"] = df["polviews_code"].apply(poli_mapping)  
        var_lst.append("political_view")
        print("political_view cleaned!")
    else:
        print("no data for political_view") 
    

    # trust
    if not df["trust"].isna().all():
        var_lst.append("trust")
        print("trust added!")
    else:
        print("no data for trust")
    

    # vocabulary test
    if not df["wordsum_code"].isna().all():
        def word_mapping(x):
            if x >= 0: 
                return x
            else:
                return np.nan
        df["vocabulary_test"] = df["wordsum_code"].apply(word_mapping)  
        var_lst.append("vocabulary_test")
        print("vocabulary_test cleaned!")
    else:
        print("no data for vocabulary_test") 
    

    # work hard
    if not df["workhard"].isna().all():
        df["work_hard"] = df["workhard"]
        var_lst.append("work_hard")
        print("work_hard added!")
    else:
        print("no data for work_hard")
    
    
    # select vars
    df = df[var_lst]
    
    # save to csv
    df.to_csv(output)
    print(f"Cleaned GSS {year} saved to {output}!")
    
    return var_lst
        

# use the function to get 2018 dataset
gss_clean(df = gss, year=2018, 
          output = "gss2018.csv")    
