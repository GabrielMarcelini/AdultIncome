"""Adult Income: auditoria, 10-fold estratificado e teste oficial reservado.
Execute: python analise.py. Todos os caminhos são relativos a este arquivo.
"""
from pathlib import Path
import json, time, hashlib, platform, zipfile, urllib.request
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'dados'; OUT=ROOT/'resultados'
SEED=42
COLS=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
NUM=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
CAT=[c for c in COLS if c not in NUM+['education','income']]
FEATURES=NUM+CAT

def metricas(y,p):
    return dict(acuracia=accuracy_score(y,p),precisao=precision_score(y,p,zero_division=0),recall=recall_score(y,p,zero_division=0),f1=f1_score(y,p,zero_division=0))

def prepare():
    return ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)
    ])

def carregar():
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    if not (DATA/'adult.data').exists():
        urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/2/adult.zip',DATA/'adult.zip')
        with zipfile.ZipFile(DATA/'adult.zip') as z:z.extractall(DATA)
    train=pd.read_csv(DATA/'adult.data',names=COLS,skipinitialspace=False)
    test=pd.read_csv(DATA/'adult.test',names=COLS,skiprows=1,skipinitialspace=False)
    raw=pd.concat([train,test],ignore_index=True)
    # Auditoria antes de substituir marcadores ou transformar atributos.
    audit={'raw_n':len(raw),'train_raw':len(train),'test_raw':len(test),'raw_target_values':raw.income.value_counts().to_dict(),'actual_nulls':raw.isna().sum().to_dict(),'raw_unique':{c:sorted(raw[c].unique().tolist()) for c in COLS}}
    audit['surrounding_spaces']={c:int((raw[c]!=raw[c].str.strip()).sum()) for c in raw.select_dtypes('object')}
    for d in (train,test):
        for c in d.select_dtypes('object'):d[c]=d[c].str.strip()
        d['income']=d.income.str.rstrip('.')
    allclean=pd.concat([train,test],ignore_index=True)
    audit['classes']=allclean.income.value_counts().to_dict()
    audit['missing_question']=(allclean=='?').sum().to_dict()
    audit['rows_missing']=int((allclean=='?').any(axis=1).sum())
    audit['exact_duplicates_all']=int(allclean.duplicated().sum())
    audit['education_one_to_one']=bool((allclean.groupby('education')['education-num'].nunique()==1).all() and (allclean.groupby('education-num')['education'].nunique()==1).all())
    audit['education_map']=allclean[['education','education-num']].drop_duplicates().sort_values('education-num').to_dict('records')
    allclean[NUM].describe().to_csv(OUT/'estatisticas_numericas.csv')
    profiles=[]
    for c in COLS:
        profiles.append({'atributo':c,'tipo':'numerico' if c in NUM else 'categorico','unicos_sem_ausentes':allclean[c].replace('?',np.nan).nunique(),'faltantes':int((allclean[c]=='?').sum()),'min':allclean[c].min() if c in NUM else '', 'max':allclean[c].max() if c in NUM else ''})
    pd.DataFrame(profiles).to_csv(OUT/'perfil_atributos.csv',index=False)
    for label,d in [('train',train),('test',test)]:
        audit[label+'_exact_removed']=int(d.duplicated().sum())
        d.drop_duplicates(inplace=True)
        # Perfis idênticos com rótulos diferentes: excluir ambos, sem escolher renda arbitrariamente.
        mask=d.duplicated(subset=FEATURES,keep=False)
        audit[label+'_conflict_rows_removed']=int(mask.sum())
        d.drop(index=d.index[mask],inplace=True)
    train_keys=pd.MultiIndex.from_frame(train[FEATURES]);test_keys=pd.MultiIndex.from_frame(test[FEATURES])
    overlap=test_keys.isin(train_keys)
    audit['test_overlap_removed']=int(overlap.sum())
    test=test.loc[~overlap].copy()
    for d in (train,test):d.replace('?',np.nan,inplace=True)
    audit['train_n']=len(train);audit['test_n']=len(test)
    audit['train_classes']=train.income.value_counts().to_dict();audit['test_classes']=test.income.value_counts().to_dict()
    audit['zero_capital']={c:int((allclean[c]==0).sum()) for c in ['capital-gain','capital-loss']}
    audit['versions']={'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}
    audit['sha256']={f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['adult.data','adult.test']}
    (OUT/'auditoria.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return train[FEATURES],(train.income=='>50K').astype(int).to_numpy(),test[FEATURES],(test.income=='>50K').astype(int).to_numpy()

def main():
    X,y,Xtest,ytest=carregar()
    models={
      'DT1':DecisionTreeClassifier(max_depth=5,random_state=SEED),
      'DT2':DecisionTreeClassifier(max_depth=12,min_samples_leaf=5,random_state=SEED),
      'DT3':DecisionTreeClassifier(random_state=SEED),
      'RF1':RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=2,n_jobs=2,random_state=SEED),
      'RF2':RandomForestClassifier(n_estimators=100,n_jobs=2,random_state=SEED),
      'RF3':RandomForestClassifier(n_estimators=100,class_weight='balanced',n_jobs=2,random_state=SEED),
      'LR1':LogisticRegression(C=.1,max_iter=2000,solver='lbfgs',random_state=SEED),
      'LR2':LogisticRegression(C=1,max_iter=2000,solver='lbfgs',random_state=SEED),
      'LR3':LogisticRegression(C=10,max_iter=2000,solver='lbfgs',random_state=SEED),
      'LR4':LogisticRegression(C=1,class_weight='balanced',max_iter=2000,solver='lbfgs',random_state=SEED),
      'BASE':DummyClassifier(strategy='most_frequent')}
    (OUT/'configuracoes.json').write_text(json.dumps({k:v.get_params() for k,v in models.items()},indent=2))
    cv=list(StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED).split(X,y))
    foldids=np.empty(len(y),dtype=int)
    for i,(_,va) in enumerate(cv):foldids[va]=i+1
    predictions=pd.DataFrame({'linha_original_train':X.index,'fold':foldids,'real':y})
    rows=[];summary=[];cms={}
    for name,est in models.items():
        start=time.perf_counter();oof=np.empty(len(y),dtype=int)
        for f,(tr,va) in enumerate(cv,1):
            pipe=Pipeline([('pre',prepare()),('model',clone(est))]);t=time.perf_counter()
            pipe.fit(X.iloc[tr],y[tr]);pred=pipe.predict(X.iloc[va]);oof[va]=pred
            rows.append({'config':name,'fold':f,**metricas(y[va],pred),'segundos':time.perf_counter()-t})
        frame=pd.DataFrame([r for r in rows if r['config']==name])
        row={'config':name,'algoritmo':{'DT':'Arvore de Decisao','RF':'Random Forest','LR':'Regressao Logistica','BA':'Baseline'}[name[:2]]}
        for m in ['acuracia','precisao','recall','f1']:row[m+'_media']=frame[m].mean();row[m+'_dp']=frame[m].std(ddof=1)
        row['segundos_total']=time.perf_counter()-start
        summary.append(row);cms[name]=confusion_matrix(y,oof,labels=[0,1]).tolist();predictions[name]=oof
        pd.DataFrame(rows).to_csv(OUT/'metricas_por_fold.csv',index=False)
        pd.DataFrame(summary).to_csv(OUT/'comparacao_cv.csv',index=False)
        print(name,{m:round(row[m+'_media'],4) for m in ['acuracia','precisao','recall','f1']},'s',round(row['segundos_total']),flush=True)
    predictions.to_csv(OUT/'predicoes_oof.csv',index=False)
    (OUT/'matrizes_cv.json').write_text(json.dumps(cms,indent=2))
    summary=pd.DataFrame(summary)
    # Regra definida antes de olhar o teste: maior F1 médio da classe >50K.
    best=summary.loc[summary.f1_media.idxmax(),'config']
    pipe=Pipeline([('pre',prepare()),('model',clone(models[best]))]);pipe.fit(X,y)
    p=pipe.predict(Xtest);testresult={'config':best,'n':len(ytest),**metricas(ytest,p),'matriz':confusion_matrix(ytest,p,labels=[0,1]).tolist(),'n_encoded_features':int(pipe.named_steps['pre'].transform(X.iloc[:1]).shape[1])}
    (OUT/'teste_final.json').write_text(json.dumps(testresult,indent=2))
    pd.DataFrame({'linha_original_test':Xtest.index,'real':ytest,'predito':p}).to_csv(OUT/'predicoes_teste.csv',index=False)
    # Gráficos reproduzidos a partir das tabelas medidas.
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for m,label in [('f1','F1 da classe >50K'),('acuracia','Acurácia')]:
        fig,ax=plt.subplots(figsize=(9,4.2));colors=['#247c86' if x.startswith('RF') else '#435d8c' if x.startswith('DT') else '#bd783c' if x.startswith('LR') else '#999999' for x in summary.config]
        ax.bar(summary.config,summary[m+'_media']*100,yerr=summary[m+'_dp']*100,color=colors,capsize=3)
        ax.set(ylabel=label+' (%)',ylim=(0,100),xlabel='Configuração');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/(m+'_cv.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,4.4));cm=np.array(testresult['matriz']);ax.imshow(cm,cmap='Blues')
    for i in range(2):
      for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=22,color='white' if cm[i,j]>cm.max()/2 else '#142538')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['≤50K','>50K'],yticklabels=['≤50K','>50K'],xlabel='Classe prevista',ylabel='Classe real',title='Teste oficial reservado · '+best);fig.tight_layout();fig.savefig(OUT/'matriz_teste.png',dpi=180);plt.close(fig)
    print('TESTE',testresult,flush=True)

if __name__=='__main__':main()
