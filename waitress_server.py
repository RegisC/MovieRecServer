from waitress import serve
import API
serve(API.app, host='0.0.0.0', port=5000)

