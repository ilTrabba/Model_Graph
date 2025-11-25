OPERAZIONI PRELIMINARI DA FARE:

   0. 🔍 Migrazione cloud per l’archiviazione dei parametri dei modelli (con annesse modifiche o adattamenti al codice)
   1.	✅ Tradurre in inglese sia la presentazione pp che la relazione progettuale di Big Data, inviare tutto a Torlone 
   2.	✅ Effettuare una pulizia iniziale e completa del codice e comprendere il funzionamento generale (cicli ripetuti, ruolo di funzioni chiave, ecc...)
   3.	Aggiustamento del codice:
      a.	✅ Integrare un bottone di eliminazione file sulla form di inserimento dei modelli
      b.	✅ Risolvere l’errore legato alle relazioni errate rimanenti che persistono nel tempo (ricordare i 2 fratelli)
      c.	✅ Usare effettivamente il nodo family per le query
      d.	✅ Integrare l’aggiornamento del centroide
      e.	✅ Capire effettivamente quali layer sono utili alla causa (config serve?)
   4.	✅ Capire come si effettua l’inserimento di un nuovo modello su Hugging Face
   5. ✅Valutazione generale del corretto flusso di esecuzione del sistema

NOTE:
   1. Come cambiare metrica utilizzata:
       a. nel Clustering (family_clustering.py)-> in find_best_family_match, basta cambiare la metrica passata quando si chiama la funzione calculate_distance
       b. in MoTHer (Tree_builder.py)-> stessa cosa, nella funzione build_mother_tree cambiare la metrica passata a  calculate_distance 
   
   

CLUSTERIZZAZIONE:

   1. Effettuare merge con main branch su github per futuri sviluppi
   2. ✅Confrontare nel dettaglio come facciamo la distanza L2 tra modelli noi e come la fanno quelli di MoTher
   3. Aggiungere campo foundations model come flag check o not nella form per modello
   4. Aggiungere campo a tendina della defaul metric da settare nella form per la fase di clustering in upload del modello (L2, cos, RMS-L2)
   5. Aggiungere un campo un po più tecnico per esperti per settare quali layer includere o meno nel calcolo della distanza durante la fase di clustering (backbone, backbone+embeddings+head)
   6. Confrontare nel dettaglio come facciamo la distanza L2 tra modelli noi e come la fanno quelli di MoTher
   7. Effettuare l’integrazione di un hash strutturale (con annesse questioni legate all’abbattimento dei costi computazionali)
   8. Realizzare una soglia adattiva e una confidence in grado di generalizzare un corretto funzionamento della fase di clustering
   9. Valutare re-clustering globale (notturno) che ammortizzi possibili errori del clustering incrementale



MOTHER:

   1. Possibile ottimizzazione della gestione della matrice delle distanze
   2. Anche in MoTher provare/testare altre metriche di distanza, magari quelle valutate durante il clustering

ULTERIORI OBIETTIVI PROGETTUALI:

   1. Eliminazione di un modello dal sistema o correzione inserimento(unica parziale soluzione a possibili errori di associazione tra modelli)
   2. Aggiungere dei nuovi campi nella form di inserimento di un modello nel sistema, aumentando così il dettaglio di una model card e allo stesso tempo facilitare potenzialmente il lavoro di un LLM. Campi da considerare: 
   3. Realizzare una vista (box view), accessibile tramite model card singola o creando una sezione dedicata, che mostri la famiglia (ad albero) del relativo modello di interesse.
   4.	Creare un modello in grado di elaborare prompt in NL e interpretarlo come query cypher, utile per cercare modelli o interrogare la knowledge generale accumulata dal sistema. Inizialmente sarà necessario concentrarsi su query semplici e poi successivamente più complesse (facenti uso di tag dedicati)
   5.	Testare in maniera un po' più rigorosa la teoria legata all’uso dei centroidi su base media aritmetica per approssimare un cluster di modelli, tutti riferiti allo stesso task. Valutare sia casi semplici che peggiori, capire in sostanza quanto il centroide funziona bene
   6.	Scelta di nuovi modelli per l’ampliamento del dataset con l’obiettivo di estendere il sistema su scala massiva
   7.	Testare modelli massivi e vedere come il sistema reagisce
   8.	Valutare l’utilizzo di nuove metriche al posto della L2 norm

ALTRO:

   1.	Rileggere e comprendere meglio i paper:
      a.	Unsupervised model tree heritage recovery
      b.	Model lakes
      c.	Paper centroidi stesso task (più di uno)
   2.	Realizzare un logo custom per il sistema
   3.	Realizzare un portale di login per admin
   4.	Family locks

OBIETTIVI NON INTEGRATI (per ovvie ragioni):

   1. ❌ Valutare l’utilizzo dell’ordinamento sorvolando problematiche legate all’uso parziale di metadati, quali i nomi dei layer, per il calcolo della distanza tra 2 modelli

