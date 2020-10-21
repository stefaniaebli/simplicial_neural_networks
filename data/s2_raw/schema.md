# Schema

Each line of the uncompressed s2-corpus-xx.gz text file represents a paper.
Each paper is described as a JSON dictionary with the following attributes.

* id: string, S2 generated research paper ID.
* title: string, Research paper title.
* paperAbstract: string, Extracted abstract of the paper.
* entities: list, Extracted list of relevant entities or topics.
* s2Url: string, URL to S2 research paper details page.
* s2PdfUrl: string, URL to PDF on S2 if available.
* pdfUrls: list, URLs related to this PDF scraped from the web.
* authors: list, List of authors with an S2 generated author ID and name.
* inCitations: list, List of S2 paper IDs which cited this paper.
* outCitations: list, List of S2 paper IDs which this paper cited.
* year: int, Year this paper was published as integer.
* venue: string, Extracted publication venue for this paper.
* journalName: string, Name of the journal that published this paper.
* journalVolume: string, The volume of the journal where this paper was published.
* journalPages: string, The pages of the journal where this paper was published.
* sources: list, Identifies papers sourced from DBLP or Medline.
* doi: string, Digital Object Identifier registered at doi.org.
* doiUrl: string, DOI link for registered objects.
* pmid: string, Unique identifier used by PubMed.
