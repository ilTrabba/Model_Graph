OPERAZIONI PRELIMINARI DA FARE:

   1.	Tradurre in inglese sia la presentazione pp che la relazione progettuale di Big Data, inviare tutto a Torlone
   2.	Effettuare una pulizia completa del codice e comprendere il funzionamento generale (cicli ripetuti, ruolo di funzioni chiave, ecc...)
   3.	Aggiustamento del codice:
      a.	Integrare un bottone di eliminazione file sulla form di inserimento dei modelli
      b.	Risolvere l’errore legato alle relazioni errate rimanenti che persistono nel tempo (ricordare l’esempio dei 2 fratelli)
      c.	Usare effettivamente il nodo family per le query
      d.	Integrare l’aggiornamento del centroide
      e.	Capire effettivamente quali layer sono utili alla causa (config serve?)
      f.	Migrazione cloud per l’archiviazione dei parametri dei modelli (con annesse modifiche o adattamenti al codice)
   4.	Capire come si effettua l’inserimento di un nuovo modello su Hugging Face

NUOVI OBIETTIVI PROGETTUALI:

   1.	Eliminazione di un modello dal sistema (unica parziale soluzione a possibili errori di associazione tra modelli)
   2.	Aggiungere dei nuovi campi nella form di inserimento di un modello nel sistema, aumentando così il dettaglio di una model card e allo stesso tempo facilitare potenzialmente il lavoro di un LLM.
   3.	Realizzare una vista (box view), accessibile tramite model card singola o creando una sezione dedicata, che mostri la famiglia (ad albero) del relativo modello di interesse.
   4.	Creare un modello in grado di elaborare prompt in NL e interpretarlo come query cypher, utile per cercare modelli o interrogare la knowledge generale accumulata dal sistema. Inizialmente sarà necessario concentrarsi su query semplici e poi successivamente più complesse (facenti uso di tag dedicati)
   5.	Valutare l’utilizzo dell’ordinamento sorvolando problematiche legate all’uso parziale di metadati, quali i nomi dei layer, per il calcolo della distanza tra 2 modelli.
   6.	Effettuare l’integrazione di un hash strutturale (con annesse questioni legate all’abbattimento dei costi computazionali).
   7.	Realizzare una soglia adattiva in grado di generalizzare un corretto funzionamento della fase di clustering
   8.	Testare in maniera un po' più rigorosa la teoria legata all’uso dei centroidi su base media aritmetica per approssimare un cluster di modelli, tutti riferiti allo stesso task. Valutare sia casi semplici che peggiori.
   9.	Scelta di nuovi modelli per l’ampliamento del dataset con l’obiettivo di estendere il sistema su scala massiva
   10.	Testare modelli massivi e vedere come il sistema reagisce
   11.	Valutare l’utilizzo di nuove metriche al posto della L2 norm

ALTRO:

   1.	Rileggere e comprendere meglio i paper:
      a.	Unsupervised model tree heritage recovery
      b.	Model lakes
      c.	Paper centroidi stesso task (più di uno)
   2.	Realizzare un logo custom per il sistema
   3.	Realizzare un portale di login per admin
   4.	Family locks

