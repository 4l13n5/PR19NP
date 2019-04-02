
# Kickstarter projects analysis

## Vmesno Poročilo

#### Nejc Pešič

## Opis problema

V okviru projektne naloge se bom ukvarjal z analizo projektov z platforme Kickstarter.

Poskusil bom odgovoriti na vprašanja kot:
    - Kakšni projekti imajo največje možnosti da uspejo
    - Kdaj in kje imajo projekti najboljše možnosti za uspeh
    - Časovna distribucija projektov skozi leta
    - Vpliv nepopolnih podatkov na uspešnost projekta
    - Z uporabo klasifikacijskih modelov bom poskusil napovedati uspešnost projektov preden se zaključijo

## Podatki

Podatke sem našel na spletni strani Kaggle

https://www.kaggle.com/kemical/kickstarter-projects

Podani so v formatu CSV in vsebujejo podatke o projektih med leti 2009 in 2018.
Vsaka vrstica vsebuje informacije o enem projektu:
    - ID projekta
    - Ime projekta
    - Glavna in podkategorija projekta
    - Valuta v kateri so se zbirale donacije
    - Datum objave in ciljni rok
    - Država iz katere projekt izvira
    - Število podpornikov
    - Količina donacij
    - Stanje projekta


```python
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import sklearn.preprocessing
data2018 = pd.read_csv("./ks-projects-201801.csv")
```

Podatke preberem v pandas DataFrame. Stolpce z podatki o datumu objave in ciljnem roku pretvorim format datetime in dodam stolpec 'duration' ki ima podatek o trajanju projekta v dnevih.


```python
data2018['launched'] = pd.to_datetime(data2018['launched'])
data2018['deadline'] = pd.to_datetime(data2018['deadline'])
data2018['duration'] = (data2018['deadline'] - data2018['launched']).dt.days

```

Najprej izrišem graf stanj projektov. Vidimo da je uspešnih približno 35% projektov, ostali so ali neuspešni ali preklicani. Imamo tudi nekaj še aktivnih projektov in nekaj projektov z nedefiniranim stanjem. Te podatke bom kasneje izločil ker se iz njih ne moremo dosti naučiti.


```python
data2018.groupby("state").size().sort_values(ascending=False).plot.bar()
plt.show()
```


![png](Vmesno%20poro%C4%8Dilo_files/Vmesno%20poro%C4%8Dilo_7_0.png)


Tukaj izrišem grafe o številu objavljenih in uspešnih projektov. Vidimo, da čeprav je v letih 2014 in 2015 bilo objavljenih veliko večje število projektov, se število uspešnih projektov skoraj ni spremenilo. Zanimiv te tudi podatek o projektih za katere piše, da so se začeli leta 1970. Čeprav je nekaj takih projektov bilo objavljenih, v grafu uspešnosti vidimo, da ni bil niti eden od njih uspešen. To si lahko razlagamo kot preprosto napako z beleženjem podatkov na Kickstarterju, mogoče je pa tudi, da ljudje nočejo podpirati projektov z čudnimi opisi.


```python
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(15,5))
data2018.groupby(data2018['launched'].map(lambda x: x.year)).size().plot.bar(ax=axes[0],title="Projects launched pre year")
data2018.loc[data2018['state'] == 'successful'].groupby(data2018['launched'].map(lambda x: x.year)).size().plot.bar(ax=axes[1],title="Projects successful per year")
plt.show()
```


![png](Vmesno%20poro%C4%8Dilo_files/Vmesno%20poro%C4%8Dilo_9_0.png)


Na spodnjem grafu je prikazan odstotek uspešnih projektov na mesec. Razvidna sta dva padca uspešnosti, poletni in decemberski.
Padca sta mogoče povezana z tem, da v tem času v letu ljudje več zapravijo za počitnice, darila etc. in posledično nimajo toliko denarja z vlaganje v Kickstarter.


```python
success_per_month = data2018.loc[data2018['state'] == 'successful'].groupby(data2018['launched'].map(lambda x: x.month)).size()
all_per_month = data2018.groupby(data2018['launched'].map(lambda x: x.month)).size()

monthly_percentage = success_per_month / all_per_month
monthly_percentage.plot.bar(title="Percentage of successful projects by month")
plt.show()
```


![png](Vmesno%20poro%C4%8Dilo_files/Vmesno%20poro%C4%8Dilo_11_0.png)


Spodaj je graf ki prikazuje uspešnost projektov glede na državo izvora. Vidimo, da so ameriški projekti najuspešnejši in italijanski najmanj uspešni. Zanimiv je zadnji stolpec v grafu, z državo ki se kliče N,0". Predvidevam, da je to oznaka, ki je dodeljena projektom brez podatka o državi izvora oziroma z neveljavnimi podatki. Po odstotku uspešnosti izgleda, da projekti z nepopolnimi podatki nimajo velikih možnosti za uspeh.


```python
all_projects = data2018.groupby("country").size().sort_values(ascending=False).to_frame()
success = data2018.loc[data2018['state'] == 'successful'].groupby("country").size().sort_values(ascending=False).to_frame()
tmp=pd.concat([all_projects,success],axis=1, join="inner")
tmp.columns = ["all","success"]
tmp['percentage'] = tmp['success'] / tmp['all']
tmp['percentage'].sort_values(ascending=False).plot.bar()
plt.title("Success percentage by country")
plt.show()
```


![png](Vmesno%20poro%C4%8Dilo_files/Vmesno%20poro%C4%8Dilo_13_0.png)


Na spodnjih grafih so izrisani podatki o kategorijah in podkategorijah projektov. Vidimo, da je daleč največ projektov povezanih z umetnostjo in zabavo. 


```python
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(15,10))
plt.subplots_adjust(hspace=1)

data2018.groupby(data2018['main_category'] ).size().sort_values(ascending=False).plot.bar(ax=axes[0,0],title="Distribution by main category")
data2018.groupby(data2018['category'] ).size().head(20).sort_values(ascending=False).plot.bar(ax=axes[0,1],title="Distribution by sub category")

data2018.loc[data2018['state'] == 'successful'].groupby(data2018['main_category'] ).size().sort_values(ascending=False).plot.bar(ax=axes[1,0],title="Success distribution by main category")
data2018.loc[data2018['state'] == 'successful'].groupby(data2018['category'] ).size().head(20).sort_values(ascending=False).plot.bar(ax=axes[1,1],title="Success distribution by sub category")

plt.show()
```


![png](Vmesno%20poro%C4%8Dilo_files/Vmesno%20poro%C4%8Dilo_15_0.png)


Na spodnjem grafu so izrisana povprečna trajanja projektov glede na stanje projekta. Vidimo, da se uspešni projekti zaključijo večinoma v 31 dneh in dlje traja projekt, manjše so možnosti za uspeh. Če pogledamo stolpec live, vidimo da so aktivni projekti v povprečju že presegli 30 dni in bojo z veliko verjetnostjo propadli.


```python
data2018.groupby(data2018['state'])['duration'].mean().to_frame().plot.bar()
plt.show()
```


![png](Vmesno%20poro%C4%8Dilo_files/Vmesno%20poro%C4%8Dilo_17_0.png)


Sedaj se lotimo priprave podatkov za izdelavo modelov. Najprej odstranimo podatke, ki očitno nimajo vpliva na uspešnost projekta, kot so id, ime in ciljni rok. Ker nas zanima samo ali bo projekt uspešen ali ne odstranimo projekte, ki so v drugih stanjih in stanje zakodiramo kot številski tip 1 ali 0, kjer 1 pomeni uspeh in 0 neuspeh. Nato nenumeričnim atributom priredimo številske vrednosti, da bomo z njimi lažje delali.


```python
data2018.drop(['ID','name','usd pledged','deadline'], axis=1, inplace=True)

data2018 = data2018[data2018.state != 'live']
data2018 = data2018[data2018.state != 'undefined']
data2018 = data2018[data2018.state != 'suspended']
data2018 = data2018[data2018.state != 'canceled']
data2018['state'] = (data2018['state'] == 'successful').astype(int)


le = sk.preprocessing.LabelEncoder()

for i in ['category','main_category','currency','country']:
    data2018[i] = le.fit_transform(data2018[i])
```

Spodaj je izrisana korelacijska matrika atributov. Vidimo, da je količina donacij močno povezana z številom podpornikov.

Stanje projekta('state') je najbolj močno odvisno količine donacij, števila podpornikov in države izvora. Vidimo tudi, da je inverzno korelirano z trajanjem projekta, kar smo zgoraj že ugotovili iz grafa.


```python
plt.figure(figsize=(10,10))
sns.heatmap(data2018.corr(),annot=True,fmt=".2f",cmap='Greens',square=True)
plt.show()
```


![png](Vmesno%20poro%C4%8Dilo_files/Vmesno%20poro%C4%8Dilo_21_0.png)


## Načrt za nadaljevanje

Zdaj ko sem se spoznazl z podatki, se lahko začnem resno ukvarjati z napovedovanjem uspešnosti projektov. Za začetek bom poiskal distribucije atributov in jih po potrebi normaliziral. 
Nato bom naredil klasifikacijske in regresijske modele, z katerimi bom ocenil uspešnost/neuspešnost projektov preden se zaključijo.
