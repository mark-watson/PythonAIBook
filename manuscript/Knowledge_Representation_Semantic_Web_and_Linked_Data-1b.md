### Extending RDF with RDF Schema

RDF Schema (RDFS) supports the definition of classes and properties based on set inclusion. In RDFS classes and properties are orthogonal. Let’s start with looking at an example using additional namespaces:

```sparql
@prefix kb:   <http://knowledgebooks.com/ontology#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dbo:  <http://dbpedia.org/ontology/> .

<http://news.com/201234/>
  kb:containsCountry
  <http://dbpedia.org/resource/China>  .
  
<http://news.com/201234/>
  kb:containsCountry
  <http://dbpedia.org/resource/United_States>  .
  
<http://dbpedia.org/resource/China>
  rdfs:label "China"@en,
  rdf:type dbo:Place ,
  rdf:type dbo:Country .
```

Because the Semantic Web is intended to be processed automatically by software systems it is encoded as RDF. There is a problem that must be solved in implementing and using the Semantic Web: everyone who publishes Semantic Web data is free to create their own RDF schemas for storing data. For example, there is usually no single standard RDF schema definition for topics like news stories and stock market data. The SKOS is a namespace containing standard schemas and the most widely used standard is schema.org. Understanding the ways of integrating different data sources using different schemas helps to understand the design decisions behind the Semantic Web applications. In this chapter I often use my own schemas in the knowledgebooks.com namespace for the simple examples you see here. When you build your own production systems part of the work is searching through schema.org and SKOS to use standard name spaces and schemas when possible because this facilitates linking your data to other RDF Data on the web. The use of standard schemas helps when you link internal proprietary Knowledge Graphs used in organization with public open data from sources like WikiData and DBPedia.

Let’s consider an example: suppose that your local Knowledge Graph referred to President Joe Biden in which case we could “mint” our own URI like:

1 https://knowledgebooks.com/person#Joe_Biden
In this case users of the local Knowledge Graph could not take advantage of connected data. For example, the DBPedia and WikiData URIs for How Biden are:

    https://dbpedia.org/resource/Joe_Biden
    http://www.wikidata.org/entity/Q6279

Both of these URIs can be followed by clicking on the links if you are reading a PDF copy of this book. Please “follow your nose” and see how both of these URIs resolve to human-readable web pages.

After telling you, dear reader, to always try to use public and standard URIs like the above examples for Joe Biden, I will now revert to using simple made-up URIs for the following discussion.

We will start with an example that is an extension of the example in the last section that also uses RDFS. We add a few additional RDF statements:

```sparql
@prefix kb:  <http://knowledgebooks.com/ontology#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .

kb:containsCity rdfs:subPropertyOf kb:containsPlace .
kb:containsCountry rdfs:subPropertyOf kb:containsPlace .
kb:containsState rdfs:subPropertyOf kb:containsPlace .
```

The last three lines declare that:

- The property containsCity is a sub-property of containsPlace.
- The property containsCountry is a sub-property of containsPlace.
- The property containsState is a sub-property of containsPlace.

Why is this useful? For at least two reasons:

You can query an RDF data store for all triples that use property containsPlace and also match triples with properties equal to containsCity, containsCountry, or containsState. There may not even be any triples that explicitly use the property containsPlace.

Consider a hypothetical case where you are using two different RDF data stores that use different properties for naming cities: cityName and city. You can define cityName to be a sub-property of city and then write all queries against the single property name city. This removes the necessity to convert data from different sources to use the same Schema. You can also use OWL to state property and class equivalency.

In addition to providing a vocabulary for describing properties and class membership by properties, RDFS is also used for logical inference to infer new triples, combine data from different RDF data sources, and to allow effective querying of RDF data stores. We will see examples of all of these features of RDFS when we later start using the Jena libraries to perform SPARQL queries.

### The SPARQL Query Language

SPARQL is a query language used to query RDF data stores. While SPARQL may initially look like SQL, we will see that there are some important differences like support for RDFS and OWL inferencing and graph-based instead of relational matching operations. We will cover the basics of SPARQL in this section and then see more examples later when we learn how to embed Jena in Java applications, and see more examples in the last chapter Knowledge Graph Navigator.

We will use the N3 format RDF file test_data/news.n3 for the examples. I created this file automatically by spidering Reuters news stories on the news.yahoo.com web site and automatically extracting named entities from the text of the articles. We saw techniques for extracting named entities from text in earlier chapters. In this chapter we use these sample RDF files.

You have already seen snippets of this file and I list the entire file here for reference, edited to fit line width: you may find the file news.n3 easier to read if you are at your computer and open the file in a text editor so you will not be limited to what fits on a book page:

```sparql
@prefix kb:  <http://knowledgebooks.com/ontology#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .

kb:containsCity rdfs:subPropertyOf kb:containsPlace .

kb:containsCountry rdfs:subPropertyOf kb:containsPlace .

kb:containsState rdfs:subPropertyOf kb:containsPlace .

<http://yahoo.com/20080616/usa_flooding_dc_16/>
        kb:containsCity "Burlington" , "Denver" ,
                        "St. Paul" ," Chicago" ,
                        "Quincy" , "CHICAGO" ,
                        "Iowa City" ;
        kb:containsRegion "U.S. Midwest" , "Midwest" ;
        kb:containsCountry "United States" , "Japan" ;
        kb:containsState "Minnesota" , "Illinois" , 
                         "Mississippi" , "Iowa" ;
        kb:containsOrganization "National Guard" ,
                         "U.S. Department of Agriculture",
                         "White House" ,
                         "Chicago Board of Trade" ,
                         "Department of Transportation" ;
        kb:containsPerson "Dena Gray-Fisher" ,
                          "Donald Miller" ,
                          "Glenn Hollander" ,
                          "Rich Feltes" ,
                          "George W. Bush" ;
        kb:containsIndustryTerm "food inflation", "food",
                                "finance ministers" ,
                                "oil" .

<http://yahoo.com/78325/ts_nm/usa_politics_dc_2/>
        kb:containsCity "Washington" , "Baghdad" ,
                        "Arlington" , "Flint" ;
        kb:containsCountry "United States" ,
                           "Afghanistan" ,
                           "Iraq" ;
        kb:containsState "Illinois" , "Virginia" ,
                         "Arizona" , "Michigan" ;
        kb:containsOrganization "White House" ,
                                "Obama administration" ,
                                "Iraqi government" ;
        kb:containsPerson "David Petraeus" ,
                          "John McCain" ,
                          "Hoshiyar Zebari" ,
                          "Barack Obama" ,
                          "George W. Bush" ,
                          "Carly Fiorina" ;
        kb:containsIndustryTerm "oil prices" .

<http://yahoo.com/10944/ts_nm/worldleaders_dc_1/>
        kb:containsCity "WASHINGTON" ;
        kb:containsCountry "United States" , "Pakistan" ,
                           "Islamic Republic of Iran" ;
        kb:containsState "Maryland" ;
        kb:containsOrganization "University of Maryland" ,
                                "United Nations" ;
        kb:containsPerson "Ban Ki-moon" , "Gordon Brown" ,
                          "Hu Jintao" , "George W. Bush" ,
                          "Pervez Musharraf" ,
                          "Vladimir Putin" ,
                          "Steven Kull" ,
                          "Mahmoud Ahmadinejad" .

<http://yahoo.com/10622/global_economy_dc_4/>
        kb:containsCity "Sao Paulo" , "Kuala Lumpur" ;
        kb:containsRegion "Midwest" ;
        kb:containsCountry "United States" , "Britain" ,
                           "Saudi Arabia" , "Spain" ,
                           "Italy" , India" , 
                           ""France" , "Canada" ,
                           "Russia", "Germany", "China",
                           "Japan" , "South Korea" ;
        kb:containsOrganization "Federal Reserve Bank" ,
                                "European Union" ,
                                "European Central Bank" ,
                                "European Commission" ;
        kb:containsPerson "Lee Myung-bak" , "Rajat Nag" ,
                          "Luiz Inacio Lula da Silva" ,
                          "Jeffrey Lacker" ;
        kb:containsCompany
            "Development Bank Managing" ,
            "Reuters" ,
            "Richmond Federal Reserve Bank";
        kb:containsIndustryTerm "central bank" , "food" ,
                                "energy costs" ,
                                "finance ministers" ,
                                "crude oil prices" ,
                                "oil prices" ,
                                "oil shock" ,
                                "food prices" ,
                                "Finance ministers" ,
                                "Oil prices" , "oil" .
```

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
