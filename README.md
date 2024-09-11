# Flask Similarity API

This app finds the similarity between two texts (sentence1 and sentence2) and returns the calculated similarity value using the cosine method.

## Initial Steps to Run

1. **Ensure you have these installed:**
 - **[python](https://www.python.org/downloads/)**

2. **Ensure you have installed the python packages:**
 - Run below commands to install needed packages.
```bash
pip install flask sentence-transformers numpy
```

3. **Test the app using the following command:**
```sh
  curl -X POST http://localhost:5000/similarity \
	  -H "Content-Type: application/json" \
	  -d '{"sentence1": "This is a sentence.", "sentence2": "This is another sentence."}'
```
# Example Response
Here’s an example of what the response might look like:
```JSON
  JSON
    {
      "similarity": 0.85
    }
```

# Additional Information
 - **License:** Follow is licensed under the GNU General Public License version 3.
## Contact
 - Feel free to reach out if you have any questions or suggestions!