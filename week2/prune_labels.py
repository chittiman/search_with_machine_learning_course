import argparse


def prune_data(src_file,tgt_file):
    lines = load_file(src_file)
    data = parse_data(lines)
    label_prod_dict = classify_products(data)
    filtered_data = filter_data(label_prod_dict)
    prepared_data_lines = prepare_data(filtered_data)
    write_file(prepared_data_lines, tgt_file)
    print (f"Total {len(filtered_data)} labels with {len(prepared_data_lines)} products written to {tgt_file}")


def load_file(file):
    with open(file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def write_file(lines,file):
    lines = [line+"\n" for line in lines]
    with open(file, "w+") as f:
        f.writelines(lines)
    print (f"Files written to {file}")

def parse_data(lines):
    lines = [line.split(" ", 1) for line in lines]
    return lines

def classify_products(data):
    label_products_dict = {}
    for (label,product) in data:
        try:
            label_products_dict[label].append(product)
        except KeyError:
            #Initiate the products list
            products = [product]
            label_products_dict[label] = products
    return label_products_dict

def filter_data(data_dict, threshold = 500):
    filtered_data = {}
    total_products = 0
    total_labels = 0
    for label,products in data_dict.items():
        if len(products) >= threshold:
            filtered_data[label] = products
            total_labels += 1
            total_products += len(products)
    print (f"Final count --> {total_products} prodcuts {total_labels} labels")
    return filtered_data

def prepare_data(data_dict):
    lines = []
    for label,products in data_dict.items():
        for product in products:
            line = f"{label} {product}"
            lines.append(line)
    return lines



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", help = "Path to the file with original label_product pairs")
    parser.add_argument("--tgt_file", help = "Path to the file with pruned label_product pairs")
    args = parser.parse_args()
    prune_data(args.src_file,args.tgt_file)
    # lines = load_file(args.src_file)
    # data = parse_data(lines)
    # label_prod_dict = classify_products(data)
    # filtered_data = filter_data(label_prod_dict)
    # print (len(label_prod_dict), len(filtered_data))
