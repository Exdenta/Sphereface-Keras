def process(subdirectory, save_dir, dataset_path):
    detector = MTCNN()
    # in case subdirectory already exists
    if not (save_dir / subdirectory).exists():
        (save_dir / subdirectory).mkdir()

    images_directory = dataset_path / subdirectory
    for image_name in os.listdir(images_directory):
        image_path = images_directory / image_name
        image = cv2.imread(str(image_path))
        result = detect_face(detector, image)
        if result == None:
            continue
        keypoints, (x1, y1), (x2, y2) = result
        image = align(image, keypoints)
        image = image[y1:y2, x1:x2]
        image = cv2.resize(image, image_size, cv2.INTER_AREA)
        cv2.imwrite(str(save_dir / subdirectory / image_name), image)

def parse_arguments():
    parser = argparse.ArgumentParser('Detect, align and save faces')
    parser.add_argument('--source_dir', help='directory with original images')
    parser.add_argument(
        '--save_dir', help='directory to save images with aligned faces')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    source_dir = pathlib.Path(args.source_dir)
    save_dir = pathlib.Path(args.save_dir)
    print("Processing {} dataset".format(source_dir))
    print("Aligned faces will be saves to {}".format(save_dir))

    if not source_dir.exists():
        exit("Path {} doesn't exist!".format(source_dir))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # each subdirectory in lwf dataset in a specific person photo collection
    subdirectories = os.listdir(source_dir)

    print("Detect and align faces:\n")
    pool = multiprocessing.Pool(processes=process_count)
    pbar = tqdm(total=len(subdirectories))

    def update(*a):
        pbar.update()
    async_requests = [pool.apply_async(process, args=(
        subdirectory, save_dir, source_dir), callback=update) for subdirectory in subdirectories]
    pool.close()
    pool.join()