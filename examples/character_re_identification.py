from argparse import ArgumentParser
import glob
import os

from animecv.util import load_image
from animecv.character_identification import Res18_CharacterIdentifier

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--target", help="Filename of target picture.")
    parser.add_argument("--reference", nargs="+", 
        help="List of directories which contains reference pictures. One directory per character.")
    parser.add_argument("--name", nargs="+",
        help="List of character names. Character order must matches to that of --reference.")
    parser.add_argument("--cuda", action="store_true",
        help="Use GPU or not.")

    args = parser.parse_args()

    # Load target picture.
    target_img = load_image(args.target)

    # Load all reference pictures.
    reference_img = [
        [load_image(fn) for fn in glob.glob(os.path.join(directory, "*"))]
        for directory in args.reference
    ]

    # Prepare pre-trained model.
    # Parameter file will be saved under ~/.animecv by default.
    identifier = Res18_CharacterIdentifier()

    # Move encoder to GPU if required.
    if args.cuda:
        identifier.to("cuda")
    
    # Encode all pictures.
    # Note that argument is a list of PIL images.
    target_emb = identifier.encode_image([target_img])
    reference_emb = [
        identifier.encode_image(lst_images)
        for lst_images in reference_img
    ]

    # Compute similarity scores between target picture and each character.
    # There are two modes, Avg and Max.
    # Avg will return average score while Max returns maximum score for each character.
    scores = identifier.identify(
        target_emb,
        reference_emb,
        mode="Avg"
    )
    # Convert score from FloatTensor to numpy array.
    scores = scores.detach().cpu().numpy()

    # Print score for each character.
    for i_character in range(scores.shape[1]):
        character_name = args.name[i_character]
        character_score = scores[0, i_character]

        print(f"Score for character {character_name} is {character_score:.2f}")



