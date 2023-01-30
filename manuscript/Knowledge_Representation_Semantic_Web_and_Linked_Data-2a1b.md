
Of particular use is the matched result:

```
?p	?o
<http://www.w3.org/2002/07/owl#sameAs>
  <http://dbpedia.org/resource/Hillary_Rodham_Clinton>
```

This lets us combine data from OpenCyc with DBPedia (limit to just 2500 results). This is not a good example since we are in no way tying together data from OpenCyc to DBPedia (we will combine the results later), rather we are just doing two separate queries:

```sparql
SELECT *
WHERE {
  <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA> ?p ?o .

  SERVICE <http://dbpedia.org/sparql?timeout=30000> {
    <http://dbpedia.org/resource/Hillary_Rodham_Clinton>
      ?p_dbpedia_1
      ?o_dbpedia_1 .
    ?s_dbpedia_2
      ?p_dbpedia_2
      <http://dbpedia.org/resource/Hillary_Rodham_Clinton> .
  }
} limit 2500
```

Two results chosen that only used English language (DBPedia has triples containing text in many human languages):
