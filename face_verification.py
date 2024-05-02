from deepface import DeepFace

def Face_verification(initial, intermediate):
    try:
        results = DeepFace.verify(img1_path = initial, img2_path = intermediate)
    except:
        return 'Error'

    return bool(results['verified'])


if __name__ == "__main__":
    initial_path = input("Enter the initial photo: \n")
    intermediate_path = input("Enter the intermediate/final photo: \n")
    results = Face_verification(initial=initial_path,intermediate=intermediate_path)
    print(results)