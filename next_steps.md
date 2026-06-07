# Next steps — Disparity filter in `get_similarity_hub_score`

## Contesto

Valutazione sulla possibilità di sostituire il filtro "naïf" di edge filtering in
[`get_similarity_hub_score`](src/astrodetection/utils.py) con il **disparity filter**
multiscala di Serrano, Boguñá & Vespignani (2009), *Extracting the multiscale backbone
of complex weighted networks* (PNAS).

## Cosa fa il filtro attuale

```python
edges_to_remove = [(u, v) for u, v, d in G_sharing.edges(data=True)
                   if d.get('weight', 0) < threshold]   # threshold = 0.9
```

È **esattamente il "global threshold filter"** che il paper usa come termine di paragone:
un taglio globale a `ωc = 0.9`, uguale per tutti i nodi. Difetti noti (dal paper):
introduce una scala caratteristica arbitraria e penalizza i nodi a bassa *strength*.

Il grafo su cui agisce (`create_coSharing_graph`) è non orientato, con pesi = similarità
coseno TF-IDF in (0, 1], già filtrato per `min_overlap` (≥3 account condivisi).

## Come funziona il disparity filter

Per ogni nodo `i` di grado `k`, normalizza i pesi incidenti `p_ij = w_ij / s_i`
(con `s_i = Σ w_ij`) e calcola un p-value rispetto a un null model uniforme.
La forma chiusa dell'Eq. [2] si semplifica a:

```
α_ij = (1 − p_ij)^(k − 1)
```

Si tiene l'arco se `α_ij < α` per **almeno uno** dei due estremi (regola OR).
Nessun integrale numerico necessario.

## Raccomandazione: NON implementarlo come sostituzione

Tenerlo al massimo come opzione sperimentale. Se si deve scegliere una direzione,
**mantenere il global threshold attuale.**

### Motivo principale — tensione semantica (dirimente)

- Il disparity filter è uno strumento per **estrarre un backbone connesso**.
  La regola OR è progettata apposta per *non* spezzare il grafo
  («ensures that small nodes are not belittled so that the system remains in the
  percolated phase»).
- `get_similarity_hub_score` fa l'opposto: vuole **frammentare** il grafo per isolare
  il nucleo coordinato e misurarne la dimensione relativa
  (`largest_community / utenti_totali`).

Si userebbe quindi uno strumento ottimizzato per uno scopo per il suo contrario.
Si può forzare con la regola AND, ma — come ammette la SI del paper stesso —
in quel caso il risultato torna ad assomigliare al global threshold
(Fig. 7: «AND disparity filter is qualitatively very similar to the global threshold...
while OR maintains a much larger number of nodes»). Cioè: o cambia poco, o cambia
nella direzione sbagliata.

### Motivi pragmatici aggiuntivi

1. **Costo/beneficio sfavorevole.** I pesi coseno sono bounded in (0, 1] e non
   heavy-tailed: è proprio il regime in cui il paper ammette che disparity ≈ threshold.
   Si investirebbe complessità (gestione `k=1`, scelta di `α`, sincronizzazione dei due
   package `astrodetection` / `astrodetection_light`) per un guadagno teorico che su
   questi dati potrebbe non materializzarsi.

2. **Perdita di interpretabilità.** "Utenti ≥90% simili" è una frase comunicabile in un
   report. "Significatività statistica α rispetto a un null uniforme" no — e questa è
   una metrica di bot-likelihood che deve restare leggibile.

3. **Assenza di test suite.** Cambiare il cuore di una metrica senza rete di sicurezza,
   su un punto dove il vantaggio è incerto, è rischio gratuito.

### Note pratiche (se mai si procedesse comunque)

- **Regola AND, non OR**: frammenta di più, più adatta allo scopo dell'hub score.
- **Nodi di grado 1**: dopo `min_overlap` ce ne sono molti; per `k=1` la formula
  degenera (`α = (1 − p)^0 = 1`, mai significativo). Vanno gestiti come caso speciale
  (arco tenuto solo se significativo per l'altro estremo); una coppia isolata
  `k=1`–`k=1` verrebbe sempre scartata.
- Implementare come funzione separata + parametro `filter_method`
  (`'threshold'` | `'disparity'`) con default invariato, sincronizzando entrambi i package.

## Quando rivedere questa decisione

Se sui dataset reali la soglia 0.9 risultasse fragile — score che oscillano molto al
variare della soglia, oppure cluster coordinati evidenti distrutti dal taglio a 0.9.

In quel caso il problema vero non è "threshold vs disparity", ma il fatto che *una soglia
globale fissa è arbitraria*. La mossa più semplice e onesta sarebbe rendere la soglia
parametrica e fare una **sensitivity analysis**, non sostituire il meccanismo.

## Riferimento

Serrano M. Á., Boguñá M., Vespignani A. (2009). *Extracting the multiscale backbone of
complex weighted networks*. PNAS. arXiv:0904.2389 — formula chiave Eq. [2];
regola OR vs AND e regime a pesi non correlati nel Supporting Information.
