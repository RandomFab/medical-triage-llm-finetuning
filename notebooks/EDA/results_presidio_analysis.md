# Nombre de lignes touchées (au moins 1 balise)
mask_answer = df['answer'].str.contains(r'<[^>]+>', regex=True)
mask_question = df['question'].str.contains(r'<[^>]+>', regex=True)
mask_any = mask_answer | mask_question

print(f"Lignes avec balises dans answer : {mask_answer.sum()} / {len(df)} ({mask_answer.mean():.1%})")
print(f"Lignes avec balises dans question : {mask_question.sum()} / {len(df)} ({mask_question.mean():.1%})")
print(f"Lignes touchées (question OU answer) : {mask_any.sum()} / {len(df)} ({mask_any.mean():.1%})")

print("\n--- Répartition par source ---")
print(df[mask_any].groupby('dataset_name').size())
print("\n--- Répartition par type de balise (answer) ---")
print(df['answer'].str.extractall(r'(<[^>]+>)')[0].value_counts())

```
Lignes avec balises dans answer : 1611 / 5000 (32.2%)
Lignes avec balises dans question : 1387 / 5000 (27.7%)
Lignes touchées (question OU answer) : 2340 / 5000 (46.8%)

--- Répartition par source ---
dataset_name
frenchmedmcqa     158
mediqal           442
medquad           616
ultramed         1124
dtype: int64

--- Répartition par type de balise (answer) ---
0
<PERSON> 1874
<DATE> 1451
<LOCATION> 729
<PHONE> 16
