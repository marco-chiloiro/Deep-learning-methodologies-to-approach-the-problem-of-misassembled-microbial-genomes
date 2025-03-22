# Deep-learning-methodologies-to-approach-the-problem-of-misassembled-microbial-genomes

Metagenomics is the study of the genetic material of microorganisms sampled directly from the envi-
ronment. This approach expands the field of study of microorganisms, adding results to those obtained
with traditional microbiological techniques. One of the main tasks of metagenomics is the reconstruc-
tion of genomes contained in environmental samples, which often contain unknown species. To do this,
the genetic content of an environmental sample is first sequenced, resulting in short DNA sequences
called “reads”. To assemble reads into longer DNA sequences, called “contigs”, in order to group them
into bins that will form the reconstructed genomes, so-called reference-free assembly tools are needed,
which assemble the reads together without prior knowledge. The goal of this thesis is to develop an
assembly quality assessment tool using deep learning to improve the quality of reconstructed genomes,
also called metagenomic assembled genomes (MAGs).
Particular attention is paid to the identification of assembly errors due to repetitions within the
genomes of different microorganisms. The starting dataset is the collection of over 1.5 million microbial
genomes hosted at the University of Trento. A synthetic dataset of reads is built by simulating the Illu-
mina sequencing process on several genomes almost perfectly complete and free from contamination.
The assembly procedure is then applied to simulated reads from previously selected high-quality genome
pairs having highly similar genomes, to increase the possibility of assembly errors due to DNA repeats
in the obtained contigs. The simulated reads are then re-mapped onto the assembly contigs, in order to
assign a degree of error per position as a function of the depth of coverage, i.e. how many reads align
at a given position along a given contig. Briefly, if a region is not completely mapped by either genome,
it is considered incorrectly assembled. Two different deep learning models, together with a benchmark
model and two ensemble models, are trained on these data to predict the presence of misassembled re-
gions in contigs, with the aim of developing a post-assembly quality control tool.
The models generalize quite well, particularly when combined into an ensemble, achieving AUC
scores of ∼ 0.8 on training-related genomes and ∼ 0.7 on external genomes. While to be optimized,
the presented models are capable of learning DNA-related properties of the genomic sequences in order
to distinguish between correctly and erroneously assembled genomic regions in bacteria.
