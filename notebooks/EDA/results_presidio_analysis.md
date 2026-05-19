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

--- Analyse question DPO ---
{
  "prompt": "J'ai un mal de gorge intense depuis 2 semaines. je me sens fatigué et j'observe des tache blanche au fond de la gorge. Quel est la maladie la plus probable ? Est ce un cas à traiter en urgence ?",
  "max_tokens": 512,
  "temperature": 0.7
}
{  "response": " pouvez-vous me conseiller un traitement ?\n\nles taches blanches que vous observez au fond de la gorge sont probablement des <LOCATION>. elles sont souvent causées par un virus, comme le rhinovirus, qui peut entraîner des infections respiratoires aiguës (iras) telles que la grippe. il est également possible que ces taches soient des <LOCATION> à cause d'un bain de bouche ou d'une infection bactérienne. \n\nun mal de gorge intense depuis 2 semaines, accompagné de fatigue, peut être un signe d'une ira, d'une <LOCATION>, ou d'une infection bactérienne. il est important de consulter un médecin pour un diagnostic précis, car certaines conditions peuvent nécessiter un traitement spécifique.\n\nun traitement en urgence n'est pas nécessaire si vous n'avez pas de signes graves de complication. pour un mal de gorge sans signe d'infection, une <LOCATION> est généralement recommandée. si vous avez des <LOCATION>, une <LOCATION> peut être prescrite en fonction du type de bactérie présente. en cas de rhinovirus, une <LOCATION> peut également être prescrite. en cas d'infection bactérienne, une <LOCATION> est généralement nécessaire.\n\nil est crucial de consulter un médecin pour un diagnostic et un traitement appropriés. un mauvais diagnostic peut entraîner des complications potentielles."}
--- Analyse question SFT ---
{
  "prompt": "J'ai un mal de gorge intense depuis 2 semaines. je me sens fatigué et j'observe des tache blanche au fond de la gorge. Quel est la maladie la plus probable ? Est ce un cas à traiter en urgence ?",
  "max_tokens": 512,
  "temperature": 0.7
}
{
  "response": " que faire ? merci d'avance.\n\nbonjour,\n\nil y a plusieurs possibilités qui peuvent expliquer tes symptômes, mais il est important de noter que je ne peux pas te conseiller en ligne. il est fortement recommandé de consulter un médecin pour une évaluation complète et un diagnostic précis.\n\n1. **pharyngite virale** : c'est une infection virale du voile de la gorge qui peut durer plusieurs semaines. elle peut être accompagnée de fièvre, douloureux de gorge, et de fatigue. le traitement dépend de l'agent infectieux (antibiotiques si le virus est bactérien, antiviraux si le virus est viral, ou aucun traitement si le virus est virale).\n\n2. **pharyngite bactérienne** : si la pharyngite est due à une bactérie, elle peut être traitée avec des antibiotiques adaptés. il est important de ne pas utiliser des antibiotiques si la cause n'est pas bactérienne pour éviter les effets indésirables et la résistance aux antibiotiques.\n\n3. **amygdalite** : c'est une inflammation des amygdales (petites structures en forme de cloche à la base du palais) qui peut être accompagnée de taches blanches. elle peut être due à des infections virales ou bactériennes et peut nécessiter des antibiotiques ou des antiviraux selon le diagnostic.\n\n4. **fibrose pulmonaire** : en raison des symptômes mentionnés, il est possible que tu aies une fibrose pulmonaire, une maladie chronique qui peut entraîner des taches blanches sur les muqueuses de la gorge. c'est un cas à traiter en urgence si nécessaire.\n\n5. **sérotonine** : malheureusement, le terme \"sérotonine\" n'est pas un diagnostic médical et pourrait être une erreur de frappe. il est important de vérifier le terme exact utilisé par ton médecin pour obtenir une réponse précise.\n\n**que faire :**\n\n- si tu as des symptômes graves (fièvre, difficultés respiratoires, douleur intense), il est crucial de consulter immédiatement un médecin ou un pédiatre.\n- si tu as des antécédents de maladies respiratoires chroniques ou si tu"
}