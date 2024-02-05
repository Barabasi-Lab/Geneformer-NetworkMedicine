# Imports attention class and all required packages
from Attentions import *

# Function for obtaining user-input arguments
def get_arguments():
    parser = argparse.ArgumentParser()

    # Type of job
    parser.add_argument('-c', '--class', dest = 'chosen_class', type = str, help = 'Attentions class to use', default = 'perturber')
    parser.add_argument('-d', '--disease', dest = 'disease', type = str, help = 'Selected disease for disease-based analysis', default = 'cardiomyopathy hypertrophic')
    parser.add_argument('-s', '--samples', dest = 'samples', type = int, help = 'Number of samples to use', default = 100)
    parser.add_argument('-l', '--layer_index', dest = 'layer_index', type = int, help = 'Layer index to extract weights from', default = -1)
    parser.add_argument('-m', '--mean_aggregation', dest = 'mean_aggregation', type = bool, help = 'Whether mean (True) or max (False) aggregation should be used', default = False)

    # Instance and total number of jobs (for batch attention downloading
    parser.add_argument('-i', '--instance', dest = 'instance', type = int, help = 'Instance', default = None)
    parser.add_argument('-t', '--total_jobs', dest = 'total_jobs', type = int, help = 'Total jobs', default = None)
    args = parser.parse_args()
    
    return args


# Main Function Runtime
if __name__ == '__main__':
    args = get_arguments()

    # Attempts to map PPI directionality - ALPHA VERSION
    if args.chosen_class == 'direction':
        new_attention = PPI_attention(mean = args.mean_aggregation) 
        new_attention.scrape_attentions(disease = 'Cardiomyopathy Hypertrophic', samples = args.samples)
        new_attention.map_direction( comparison = 'default')
    
    # Performs pretrained-fine-tuned Geneformer comparison
    elif args.chosen_class.lower() == 'compare':
        new_attention = PPI_attention(mean = args.mean_aggregation)

        if args.disease == 'covid':
        
            new_attention.map_attention_changes(samples = args.samples, disease = 'covid', 
                                            keyword = f'COVID_comparison_{args.mean_aggregation}',
                                            comparison_type = 'covid', layer_index = args.layer_index)

        elif args.disease == 'cardiomyopathy hypertrophic':
            
            new_attention.map_attention_changes(samples = args.samples, disease = 'cardiomyopathy hypertrophic', 
                                            keyword = f'hcm_{args.mean_aggregation}',
                                            comparison_type = 'cardiomyopathy', layer_index = args.layer_index)
    
        elif args.disease == 'cardiomyopathy dilated':
    
            new_attention.map_attention_changes(samples = args.samples, disease = 'cardiomyopathy dilated', 
                                            keyword = f'dcm_{args.mean_aggregation}',
                                            comparison_type = 'cardiomyopathy', layer_index = args.layer_index)

        elif args.disease == 'adenocarcinoma':
       
            new_attention.map_attention_changes(samples = args.samples, disease = 'adenocarcinoma', keyword = f'carcinoma_{args.mean_aggregation}',
                                            comparison_type = 'carcinoma', layer_index = args.layer_index)
        elif args.disease == 'arthritis':
 
            new_attention.map_attention_changes(samples = args.samples, disease = 'arthritis rheumatoid', 
                                                keyword = f'rheumatoid_{args.mean_aggregation}', layer_index = args.layer_index,
                                                comparison_type = 'arthritis')
        else:
            print(f'{args.disease} is not a valid disease')

    # Routine for merging attention dictionaries of attention weights
    elif args.chosen_class == 'merge':
        new_attention = PPI_attention(mean = args.mean_aggregation) 
        new_attention.merge_attentions(limit = 10, normalize = False, 
                                       scale = False, 
                                       attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Max_attentions'))

    # Routine for calculating finetuned model on a specific disease
    elif args.chosen_class == 'disease':

        # Cardiomyopathy Hypertrophic
        if args.disease == 'cardiomyopathy hypertrophic':
            new_attention = PPI_attention(layer_index = args.layer_index, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            mean = args.mean_aggregation, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_attentions(samples = args.samples, disease = 'cardiomyopathy hypertrophic', filter_label = ('disease', 'hcm'))
            new_attention.map_disease_genes(keyword = f'finetuned_{args.mean_aggregation}_hcm', disease = 'cardiomyopathy hypertrophic')

        elif args.disease == 'cardiomyopathy dilated':

            new_attention = PPI_attention(layer_index = args.layer_index, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            mean = args.mean_aggregation, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"),)
            new_attention.scrape_attentions(samples = args.samples, disease = 'Cardiomyopathy Dilated', filter_label = ('disease', 'dcm'))
            new_attention.map_disease_genes(disease = 'Cardiomyopathy Dilated', keyword = f'disease_dcm_{args.mean_aggregation}',)
        
        elif args.disease == 'arthritis':
            new_attention = PPI_attention(layer_index = args.layer_index, mean = args.mean_aggregation, 
                                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/Arthritis_small"),)
            new_attention.scrape_attentions(samples = args.samples, disease = 'arthritis rheumatoid',)
            new_attention.map_disease_genes(disease = 'arthritis rheumatoid', keyword = f'arthritis_{args.mean_aggregation}')

        elif args.disease == 'covid':
    
            new_attention = PPI_attention(mean = args.mean_aggregation, layer_index = args.layer_index, 
                                          model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/COVID_model"),)
            new_attention.scrape_attentions(samples = args.samples, disease = 'covid',)
            new_attention.map_disease_genes(disease = 'covid', keyword = f'covid_{args.mean_aggregation}')

        elif args.disease == 'adenocarcinoma':
        
            new_attention = PPI_attention(layer_index = args.layer_index,
                                           model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/carcinoma_model"), 
                                          mean = args.mean_aggregation)
            new_attention.scrape_attentions(samples = args.samples, disease = 'adenocarcinoma',)
            new_attention.map_disease_genes(disease = 'adenocarcinoma', keyword = f'adenocarcinoma_{args.mean_aggregation}')

    # Routine for calculating general attentions from pretrained model
    else:

        # Single-shot PPI attention
        if args.instance == None:
            new_attention = PPI_attention(layer_index = args.layer_index, mean = args.mean_aggregation,)
                #dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")) # attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
            new_attention.scrape_attentions(samples = args.samples, disease = None)

            if args.disease == 'arthritis':
                new_attention.map_disease_genes(keyword = f'pretrained_arthritis_{args.mean_aggregation}', 
                                                disease = 'arthritis rheumatoid',)
                
            elif args.disease == 'cardiomyopathy hypertrophic':
                new_attention.map_disease_genes(keyword = f'pretrained_hcm_{args.mean_aggregation}', 
                                                disease = 'cardiomyopathy hypertrophic',)
            elif args.disease == 'cardiomyopathy dilated':
                new_attention.map_disease_genes(keyword = f'pretrained_dcm_{args.mean_aggregation}', 
                                                disease = 'Cardiomyopathy Dilated',)
            elif args.disease == 'covid':
                new_attention.map_disease_genes(keyword = f'pretrained_covid_{args.mean_aggregation}', 
                                                disease = 'covid',)
                
            elif args.disease == 'adenocarcinoma':
                new_attention.map_disease_genes(keyword = f'pretrained_adenocarcinoma_{args.mean_aggregation}', 
                                                disease = 'adenocarcinoma',)

            else:
                new_attention.map_PPI_genes(keyword = f'pretrained_{args.mean_aggregation}',)

        else:
            # Obtains a batch of PPI attentions
            new_attention = PPI_attention(mean = self.mean_aggregation,)
            new_attention.scrape_subset(total_jobs = args.total_jobs,
                                         instance = args.instance,)
                                         #disease = 'Cardiomyopathy Hypertrophic',
                                         #model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                                         #dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
    
    
        
