The scripts we used for manually imputed the data for the Fragile Families Challenge

Example commands:
python preprocess.py -d ffc_data/original/background.csv -t codebook_notes/aggregate_year5.tsv -m imputation_notes/aggregate_year5.tsv -v -o ffc_data/imputed/aggregate_year5_unrestricted_nostrings.csv -sr
python impute_codebook.py -c ffc_documentation/ff_teacher_cb5.txt -i codebook_notes/ff_teacher_cb5.tsv -o imputation_notes/ff_teacher_cb5.tsv -d ffc_data/original/background.csv
python curate_codebook.py -i ffc_documentation/ff_hv_cb5.txt -o codebook_notes/ff_hv_cb5.txt