import os
import re
import glob
import numpy as np
import argparse
import shutil

def merge_files_for_directory(base_dir, dir_name, selected_ys, pattern_prefix):
    """
    Traite un dossier identifié par le pattern (qui doit contenir le suffixe XXX à extraire).
    Crée un dossier full_data (vide ou le vide s'il existe déjà) et pour chaque valeur Y (filtré par selected_ys si fourni)
    génère le fichier "XXX_Y_full_data.npy" correspondant.
    """
    # Extraction de la valeur XXX depuis le nom de dossier basé sur le pattern_prefix
    m = re.search(re.escape(pattern_prefix) + r'(\d+)$', dir_name)
    if not m:
        print(f"Le dossier {dir_name} ne correspond pas au pattern attendu ({pattern_prefix}XXX).")
        return []
    val_xxx = m.group(1)
    
    # Chemins vers les dossiers des fichiers .npy
    dir_path = os.path.join(base_dir, dir_name)
    desc_path = os.path.join(dir_path, "defect_results", "AE_npy_files", "desc")
    recon_path = os.path.join(dir_path, "defect_results", "AE_npy_files", "recon_error")
    
    # Vérifier l'existence des dossiers
    if not os.path.isdir(desc_path) or not os.path.isdir(recon_path):
        print(f"Les dossiers 'desc' ou 'recon_error' n'existent pas dans {dir_name}.")
        return []
    
    # Création (ou vidage) du dossier full_data
    full_data_path = os.path.join(dir_path, "full_data")
    if os.path.exists(full_data_path):
        if os.listdir(full_data_path):
            print(f"Le dossier {full_data_path} n'est pas vide, on le vide.")
            shutil.rmtree(full_data_path)
            os.makedirs(full_data_path)
    else:
        os.makedirs(full_data_path)
    
    # Récupération de toutes les valeurs Y depuis le dossier desc
    pattern_desc = os.path.join(desc_path, "detected_defects_AE_*_desc.npy")
    desc_files = glob.glob(pattern_desc)
    ys_found = set()
    for file in desc_files:
        m = re.search(r"detected_defects_AE_(.+?)_desc\.npy$", os.path.basename(file))
        if m:
            ys_found.add(m.group(1))
    ys_found = list(ys_found)
    ys_found.sort()
    
    # Si l'utilisateur a spécifié des valeurs Y, on filtre
    if selected_ys:
        ys_to_process = [y for y in ys_found if y in selected_ys]
    else:
        ys_to_process = ys_found

    processed_ys = []
    for y in ys_to_process:
        # Chemins des fichiers pour la valeur Y
        desc_file = os.path.join(desc_path, f"detected_defects_AE_{y}_desc.npy")
        recon_file = os.path.join(recon_path, f"detected_defects_AE_{y}_.npy")
        if not os.path.exists(desc_file):
            print(f"Fichier desc non trouvé pour Y={y} dans {dir_name}.")
            continue
        if not os.path.exists(recon_file):
            print(f"Fichier recon_error non trouvé pour Y={y} dans {dir_name}.")
            continue
        
        try:
            arr_desc = np.load(desc_file)
            arr_recon = np.load(recon_file)
        except Exception as e:
            print(f"Erreur de chargement pour Y={y} dans {dir_name}: {e}")
            continue
        
        if arr_desc.shape[0] != arr_recon.shape[0]:
            print(f"Nombre de lignes différent pour Y={y} dans {dir_name}: {arr_desc.shape[0]} vs {arr_recon.shape[0]}")
            continue
        
        n = arr_desc.shape[0]
        col_xxx = np.full((n, 1), int(val_xxx))
        full_array = np.concatenate([arr_desc, col_xxx, arr_recon], axis=1)
        
        output_file = os.path.join(full_data_path, f"{val_xxx}_{y}_full_data.npy")
        np.save(output_file, full_array)
        print(f"Création de {output_file} (shape: {full_array.shape})")
        processed_ys.append(y)
    
    return processed_ys

def create_global_datasets(base_dir, dataset_dir, all_ys, pattern_prefix):
    """
    Pour chaque valeur Y, concatène tous les fichiers XXX_Y_full_data.npy (issus de chaque dossier)
    et sauvegarde le résultat dans dataset_dir avec le nom Y_full_set.npy.
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    pattern_dir = os.path.join(base_dir, pattern_prefix + "*")
    all_dirs = glob.glob(pattern_dir)
    
    for y in all_ys:
        files_to_concat = []
        for d in all_dirs:
            full_data_dir = os.path.join(d, "full_data")
            if not os.path.exists(full_data_dir):
                continue
            pattern_file = os.path.join(full_data_dir, f"*_{y}_full_data.npy")
            matched_files = glob.glob(pattern_file)
            if matched_files:
                files_to_concat.extend(matched_files)
        if not files_to_concat:
            print(f"Aucun fichier trouvé pour Y={y} dans tous les dossiers.")
            continue
        arrays = []
        for f in files_to_concat:
            try:
                arr = np.load(f)
                arrays.append(arr)
            except Exception as e:
                print(f"Erreur lors du chargement du fichier {f}: {e}")
        if arrays:
            global_array = np.concatenate(arrays, axis=0)

            dataset_dir_r = os.path.join(dataset_dir, f"recon_{y}")
            if not os.path.exists(dataset_dir_r):
                os.makedirs(dataset_dir_r)

            output_global = os.path.join(dataset_dir_r, f"{y}_full_set.npy")

            np.save(output_global, global_array)
            print(f"Création de {output_global} (shape: {global_array.shape})")
        else:
            print(f"Aucune donnée valide pour Y={y}.")

def main():
    parser = argparse.ArgumentParser(description="Fusionner les fichiers .npy pour créer des ensembles complets.")
    parser.add_argument("--base_dir", type=str, default="./run/data", help="Chemin vers le dossier de base contenant les dossiers à traiter")
    parser.add_argument("--pattern", type=str, default="ni10cr20V_NiFeCr_stoller_100k_bx98_80kev_test1_rnd_", help="Pattern commun aux dossiers (ex: ni10cr20V_NiFeCr_stoller_100k_bx98_80kev_test1_rnd_)")
    parser.add_argument("--y", nargs="*", help="Valeur(s) Y à traiter. Si non spécifié, toutes les valeurs trouvées seront traitées.")
    args = parser.parse_args()
    
    base_dir = args.base_dir
    dataset_dir = os.path.join(base_dir, "dataset")
    
    pattern_dir = os.path.join(base_dir, args.pattern + "*")
    all_dirs = glob.glob(pattern_dir)
    
    if not all_dirs:
        print("Aucun dossier correspondant n'a été trouvé dans", base_dir)
        return
    
    all_ys_global = set()
    for d in all_dirs:
        dir_name = os.path.basename(d)
        processed_ys = merge_files_for_directory(base_dir, dir_name, args.y, args.pattern)
        all_ys_global.update(processed_ys)
    
    if all_ys_global:
        all_ys_global = sorted(list(all_ys_global))
        print("Création des fichiers globaux pour les valeurs Y:", all_ys_global)
        create_global_datasets(base_dir, dataset_dir, all_ys_global, args.pattern)
    else:
        print("Aucune valeur Y n'a été traitée.")

if __name__ == "__main__":
    main()
