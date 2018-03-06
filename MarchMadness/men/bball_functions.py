import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score
from sklearn.utils import resample 

#Compile deeper game stats and aggreate them to by team means and standard deviations
def team_stats(df_all,trail=5):
    #Possessions
    df_all = df_all.loc[df_all.Season>=(df_all.Season.max()-trail)]
    w_pos = df_all.apply(lambda row: 0.96*(row.WFGA + row.WTO + 0.44*row.WFTA - row.WOR), axis=1)
    l_pos = df_all.apply(lambda row: 0.96*(row.LFGA + row.LTO + 0.44*row.LFTA - row.LOR), axis=1)
    df_all['Pos'] = (w_pos+l_pos)/2
    
    #Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
    df_all['WOffRtg'] = df_all.apply(lambda row: 100 * (row.WScore / row.Pos), axis=1)
    df_all['LOffRtg'] = df_all.apply(lambda row: 100 * (row.LScore / row.Pos), axis=1)
    #Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
    df_all['WDefRtg'] = df_all.LOffRtg
    df_all['LDefRtg'] = df_all.WOffRtg
    #Net Rating = Off.Rtg - Def.Rtg
    df_all['WNetRtg'] = df_all.apply(lambda row:(row.WOffRtg - row.WDefRtg), axis=1)
    df_all['LNetRtg'] = df_all.apply(lambda row:(row.LOffRtg - row.LDefRtg), axis=1)
                             
    #Assist Ratio : Percentage of team possessions that end in assists
    df_all['WAstR'] = df_all.apply(lambda row: 100 * row.WAst / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
    df_all['LAstR'] = df_all.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
    #Turnover Ratio: Number of turnovers of a team per 100 possessions used.
    #(TO * 100) / (FGA + (FTA * 0.44) + AST + TO)
    df_all['WTOR'] = df_all.apply(lambda row: 100 * row.WTO / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
    df_all['LTOR'] = df_all.apply(lambda row: 100 * row.LTO / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
                        
    #The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
    df_all['WTSP'] = df_all.apply(lambda row: 100 * row.WScore / (2 * (row.WFGA + 0.44 * row.WFTA)), axis=1)
    df_all['LTSP'] = df_all.apply(lambda row: 100 * row.LScore / (2 * (row.LFGA + 0.44 * row.LFTA)), axis=1)
    #eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable 
    df_all['WeFGP'] = df_all.apply(lambda row:(row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)      
    df_all['LeFGP'] = df_all.apply(lambda row:(row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)   
    #FTA Rate : How good a team is at drawing fouls.
    df_all['WFTAR'] = df_all.apply(lambda row: row.WFTA / row.WFGA, axis=1)
    df_all['LFTAR'] = df_all.apply(lambda row: row.LFTA / row.LFGA, axis=1)
                             
    #OREB% : Percentage of team offensive rebounds
    df_all['WORP'] = df_all.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
    df_all['LORP'] = df_all.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)
    #DREB% : Percentage of team defensive rebounds
    df_all['WDRP'] = df_all.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
    df_all['LDRP'] = df_all.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)                                      
    #REB% : Percentage of team total rebounds
    df_all['WRP'] = df_all.apply(lambda row: (row.WDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)
    df_all['LRP'] = df_all.apply(lambda row: (row.LDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1) 
    
    w_keeps =  [col for col in df_all.columns if 'W' in col]
    l_keeps =  [col for col in df_all.columns if 'L' in col]
    new_names = [name[1:] for name in w_keeps]
    #new_names == [name[1:] for name in l_keeps] #check that sets are identical
    w_tour = df_all
    w_tour = w_tour[w_keeps]
    w_tour.columns = new_names
    l_tour = df_all
    l_tour = l_tour[l_keeps]
    l_tour.columns = new_names
    stack_tour = pd.concat([w_tour,l_tour])
    team_means = stack_tour.groupby(['TeamID']).mean()
    team_means.columns = [col+'_mean' for col in team_means]
    team_stds = stack_tour.groupby(['TeamID']).std()
    team_stds.columns = [col+'_std' for col in team_stds.columns]
    team_stats = pd.merge(team_means,team_stds,how='left',left_index=True,right_index=True)
    return(team_stats)

def seed_to_int(seed):
        s_int = int(seed[1:3])
        return s_int

#take historical team level statistics and current seeding; combine into a dataset
def build_data(df_base,df_seeds,team_stats):
    df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
    df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
    df_seeds

    teams = tuple(team_stats.index)
    df_base = df_base.loc[df_base.WTeamID.isin(teams),]
    df_base = df_base.loc[df_base.LTeamID.isin(teams),]
    df_base['ScoreDiff'] = df_base.loc[:,'WScore']-df_base.loc[:,'LScore']
    df_base.drop(labels=['WScore','LScore'], inplace=True, axis=1)
    df_wins = df_base.loc[:,['WTeamID','LTeamID','Season']]
    df_wins = df_wins.reset_index().merge(df_seeds,how='left',left_on=['WTeamID','Season'],right_on=['TeamID','Season']).set_index('index')
    df_wins.drop(labels=['TeamID'], inplace=True, axis=1)    
    df_wins.loc[:,'seed_int'] = df_wins.apply(lambda row: row.seed_int if np.isfinite(row.seed_int) else 20, axis=1)
    df_wins = df_wins.reset_index().merge(team_stats,how='inner',left_on=['WTeamID'],right_index=True).set_index('index')
    df_w_names = ['W'+i for i in df_wins.columns]
    df_wins.columns = df_w_names

    df_losses = df_base.loc[:,['WTeamID','LTeamID','Season']]
    df_losses = df_losses.reset_index().merge(df_seeds,how='left',left_on=['LTeamID','Season'],right_on=['TeamID','Season']).set_index('index')
    df_losses.drop(labels=['TeamID'], inplace=True, axis=1)    
    df_losses.loc[:,'seed_int'] = df_losses.apply(lambda row: row.seed_int if np.isfinite(row.seed_int) else 20, axis=1)
    df_losses = df_losses.reset_index().merge(team_stats,how='inner',left_on=['LTeamID'],right_index=True).set_index('index')
    df_l_names = ['L'+i for i in df_losses.columns]
    df_losses.columns = df_l_names
    
    df_dummy = pd.merge(left=df_base, right=df_wins, how='left', left_index=True,right_index=True)
    df_concat = pd.merge(left=df_dummy, how='left',right=df_losses, left_index=True,right_index=True)
    df_concat['Seeded'] = df_concat.apply(lambda row: True if max(row.Wseed_int,row.Lseed_int) < 20 else False, axis=1)+0
    df_concat['SeedDiff'] = df_concat.Wseed_int - df_concat.Lseed_int
    for i in [x for x in team_stats.columns if '_mean' in x]:
        df_concat[i+'Diff'] = df_concat['W'+i]-df_concat['L'+i]
        df_concat.drop(labels=['W'+i,'L'+i],inplace=True,axis=1)
    for i in [x for x in team_stats.columns if '_std' in x]:
    #    df_concat[i+'Avg'] = ((df_concat['W'+i]+df_concat['L'+i])/2).combine_first(df_concat['W'+i]).combine_first(df_concat['L'+i]).combine_first(df_concat['L'+i].isna()+0)
        df_concat.drop(labels=['W'+i,'L'+i],inplace=True,axis=1)
    df_concat.drop(labels=['Season','WSeason','LSeason','Wseed_int','Lseed_int','WWTeamID','WLTeamID','LWTeamID','LLTeamID'],inplace=True,axis=1)
    
    df_predictions = df_concat
    df_predictions_inv = df_predictions*-1
    df_predictions_inv.loc[:,'Seeded'] = df_predictions_inv.Seeded*-1
    df_predictions = pd.concat([df_predictions,df_predictions_inv],axis=0)
    return(df_predictions)

#differently shaped output for final scoring
def generate_output_df(df_sample_sub,dfs,df_teams_fin):
    n_test_games = len(df_sample_sub)
    df_seeds = dfs.copy()
    
    df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
    df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
    
    output = pd.DataFrame
    output = pd.DataFrame()
    
    output['year'] = df_sample_sub.apply(lambda row: (int(row.ID.split('_')[0])), axis=1)
    output['Team1'] = df_sample_sub.apply(lambda row: (int(row.ID.split('_')[1])), axis=1)
    output['Team2'] = df_sample_sub.apply(lambda row: (int(row.ID.split('_')[2])), axis=1)
    
    output = pd.merge(left=output,right=df_seeds,how='left',left_on=['Team1','year'],right_on=['TeamID','Season'])
    output.drop(labels=['TeamID','Season'],inplace=True,axis=1)
    names = list(output.columns[:3])
    names.append('T1Seed')
    output.columns = names
    
    output = pd.merge(left=output,right=df_seeds,how='left',left_on=['Team2','year'],right_on=['TeamID','Season'])
    output.drop(labels=['TeamID','Season'],inplace=True,axis=1)
    names = list(output.columns[:4])
    names.append('T2Seed')
    output.columns = names
    
    output['SeedDiff'] = output.loc[:,'T1Seed']-output.loc[:,'T2Seed']
    output['Seeded'] = 1
    output.drop(labels=['T1Seed','T2Seed'],inplace=True,axis=1)
    
    output_cut = output.loc[:,['year','Team1','Team2']]
    output_1 = pd.merge(left=output_cut,right=df_teams_fin,how='inner',left_on=['Team1'],right_index=True)
    o1_names = [i+'1' for i in output_1.columns]
    output_1.columns = o1_names
    
    output_2 = pd.merge(left=output_cut,right=df_teams_fin,how='inner',left_on=['Team2'],right_index=True)
    o2_names = [i+'2' for i in output_2.columns]
    output_2.columns = o2_names
    
    output = pd.merge(left=output, right=output_1, how='left', left_on=['year', 'Team1','Team2'],right_on=['year1','Team11','Team21'])
    output.drop(labels=['year1','Team11','Team21'],inplace=True,axis=1)
    output = pd.merge(left=output, right=output_2, how='left', left_on=['year', 'Team1','Team2'],right_on=['year2','Team12','Team22'])
    output.drop(labels=['year2','Team12','Team22'],inplace=True,axis=1)
    
    for i in [x for x in df_teams_fin.columns if '_mean' in x]:
        output[i+'Diff'] = output[i+'1']-output[i+'2']
        output.drop(labels=[i+'1',i+'2'],inplace=True,axis=1)
    for i in [x for x in df_teams_fin.columns if '_std' in x]:
    #    output[i+'Avg'] = ((output[i+'1']+output[i+'2'])/2).combine_first(output[i+'1']).combine_first(output[i+'2']).combine_first(output[i+'2'].isna()+0)
        output.drop(labels=[i+'1',i+'2'],inplace=True,axis=1)
    return(output)

#we predict score differential because just looking at game outcome leaves a lot of info on the table
#to get a probability estimate we do 1000 boostrap iterations and take the average outcome
def bootstrap(dataset,mod_type='linear',iterations=1000,n_size=None):
    if(n_size is None):
        n_size = dataset.shape[0]
    models = list()
    scores = list()
    score2s = list()
    for i in range(iterations):
        train = resample(dataset, n_samples=n_size)
        test = dataset.loc[~dataset.index.isin(set(train.index)),]
        
        train_x = train.iloc[:,3:]
        train_y = train.loc[:,'ScoreDiff']

        test_x = test.iloc[:,3:]
        test_y = test.loc[:,'ScoreDiff']

        if(mod_type=='linear'): 
        	model = LinearRegression()
        if(mod_type=='Ridge'): 
        	model = LinearRegression()
        elif(mod_type=='gbm'):
        	params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
        	model = GradientBoostingRegressor(**params)
        else:
        	ValueError("Need to use specified model type")
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        results = [(x>0)+0 for x in predictions]
        real_results = [(x>0)+0 for x in test_y]
        score = log_loss(results, predictions,labels=[0,1])
        score2 = r2_score(predictions,test_y)
        models.append(model)
        scores.append(score)
        score2s.append(score2)
    return(models,scores,score2s)

#for better testing of actual model functionality, we use prior seasons to score a "current one"
def year_score(dfa,dfs,year,mod_type='linear',iterations=1000,n_size=None):
    df_teams_hist = team_stats(dfa.loc[dfa.Season<=(year-1)].copy())
    df_train_hist = build_data(dfa.loc[dfa.Season<=(year-1),['Season','WTeamID','LTeamID','WScore','LScore']].copy(),
                                               dfs.loc[dfs.Season<=(year-1)].copy(),
                                               df_teams_hist)
    models, scores, score2s = bootstrap(df_train_hist)
    
    df_teams_mod = team_stats(dfa.loc[dfa.Season<=year].copy())
    df_test_mod = build_data(dfa.loc[dfa.Season==year,['Season','WTeamID','LTeamID','WScore','LScore']].copy(),
                                               dfs.loc[dfs.Season==year,].copy(),
                                               df_teams_mod)
    
    predictions = np.vstack([m.predict(df_test_mod.iloc[:,3:]) for m in models])
    predictions = (predictions>0)+0
    total_predictions = predictions.mean(axis=0)
    total_predictions = np.clip(total_predictions, 0.02, 0.98)
    score = log_loss((df_test_mod.iloc[:,2]>0)+0, total_predictions,labels=[0,1])
    return(score)
