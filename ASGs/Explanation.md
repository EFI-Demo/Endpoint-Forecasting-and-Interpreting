All the dot files are ASGs extracted from CTIs using EFI. 
These CTIs were first text filtered and then ASGs were extracted. 
If the number of texts or nodes is too small, they are filtered and finally 1911 ASGs are obtained. 
These ASGs are dot files and can be converted to PDF using graphviz for display. (dot -Tpdf -o inputfile outputfile)
Due to the node read-in bug in graphviz, MD5 processing was performed on all nodes with attribute tags (S_F0/1/2/3_P_R).
