Please note that in the above RDF listing I took advantage of the free form syntax of N3 and Turtle RDF formats to reformat the data to fit page width.

In the following examples, I used the library developed in the next chapter that allows us to load multiple RDF input files and then to use SPARQL queries.

We will start with a simple SPARQL query for subjects (news article URLs) and objects (matching countries) with the value for the predicate equal to containsCountry. Variables in queries start with a question mark character and can have any names:

```sparql
SELECT ?subject ?object
  WHERE {
   ?subject
   <http://knowledgebooks.com/ontology#containsCountry>
   ?object .
}
```

It is important for you to understand what is happening when we apply the last SPARQL query to our sample data. Conceptually, all the triples in the sample data are scanned, keeping the ones where the predicate part of a triple is equal to

    http://knowledgebooks.com/ontology#containsCountry.

In practice RDF data stores supporting SPARQL queries index RDF data so a complete scan of the sample data is not required. This is analogous to relational databases where indices are created to avoid needing to perform complete scans of database tables.

In practice, when you are exploring a Knowledge Graph like DBPedia or WikiData (that are just very large collections of RDF triples), you might run a query and discover a useful or interesting entity URI in the triple store, then drill down to find out more about the entity. In a later chapter Knowledge Graph Navigator we attempt to automate this exploration process using the DBPedia data as a Knowledge Graph.

We will be using the same code to access the small example of RDF statements in our sample data as we will for accessing DBPedia or WikiData.

We can make this last query easier to read and reduce the chance of misspelling errors by using a namespace prefix:

```sparql
PREFIX kb:  <http://knowledgebooks.com/ontology#>
SELECT ?subject ?object
  WHERE {
      ?subject kb:containsCountry ?object .
  }
```

We could have filtered on any other predicate, for instance containsPlace. Here is another example using a match against a string literal to find all articles exactly matching the text “Maryland.”

```sparql
PREFIX kb:  <http://knowledgebooks.com/ontology#>
SELECT ?subject WHERE { ?subject kb:containsState "Maryland" . }
```

The output is:

    http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/


We can also match partial string literals against regular expressions:

```sparql
PREFIX kb: <http://knowledgebooks.com/ontology#>
SELECT ?subject ?object
       WHERE {
         ?subject
         kb:containsOrganization
         ?object FILTER regex(?object, "University") .
       }
```

The output is:

```
http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1
  "University of Maryland"
```

We might want to return all triples matching a property of containing an organization and where the object is a string containing the substring “University.” The matching statement after the FILTER check matches every triple that matches the subject in the first pattern:

```sparql
PREFIX kb: <http://knowledgebooks.com/ontology#>
SELECT DISTINCT ?subject ?a_predicate ?an_object
 WHERE {
    ?subject kb:containsOrganization ?object .
       FILTER regex(?object,"University") .
    ?subject ?a_predicate ?an_object .
}
ORDER BY ?a_predicate ?an_object
LIMIT 10
OFFSET 5
```

When WHERE clauses contain more than one triple pattern to match, this is equivalent to a Boolean “and” operation. The DISTINCT clause removes duplicate results. The ORDER BY clause sorts the output in alphabetical order: in this case first by predicate (containsCity, containsCountry, etc.) and then by object. The LIMIT modifier limits the number of results returned and the OFFSET modifier sets the number of matching results to skip.

The output is:

 ```
 http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
 	 http://knowledgebooks.com/ontology#containsOrganization
    "University of Maryland" .
 
http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsPerson,
	"Ban Ki-moon" .
 	 
http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsPerson
	"George W. Bush" .

http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsPerson
	"Gordon Brown" .

http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsPerson
    "Hu Jintao" .

http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
    http://knowledgebooks.com/ontology#containsPerson
    "Mahmoud Ahmadinejad" .
 
http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsPerson
	"Pervez Musharraf" .

http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsPerson
 	"Steven Kull" .

http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsPerson
    "Vladimir Putin" .

http://news.yahoo.com/s/nm/20080616/ts_nm/worldleaders_trust_dc_1/
	http://knowledgebooks.com/ontology#containsState
	"Maryland" .
```

We are finished with our quick tutorial on using the SELECT query form. There are three other query forms that I am not covering in this chapter:

- CONSTRUCT – returns a new RDF graph of query results.
- ASK – returns Boolean true or false indicating if a query matches any triples.
- DESCRIBE – returns a new RDF graph containing matched resources.
- OPTIONAL - contains patterns that do not have to match.
