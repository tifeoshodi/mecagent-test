import requests, base64, trimesh

tree = {
    'steps': [
        {'operation': 'workplane', 'params': {'plane': 'XZ'}},
        {'operation': 'polyline', 'params': {'points': [[10, -20], [10, 20], [40, 20], [40, 12.6], [38, 12.6], [30, 10], [38, 7.4], [40, 7.4], [40, -7.4], [38, -7.4], [30, -10], [38, -12.6], [40, -12.6], [40, -20], [10, -20]]}},
        {'operation': 'revolve', 'params': {'angle': 360, 'axis': 'Y'}}
    ]
}

r = requests.post('http://localhost:8000/api/compile', json=tree)
print('Status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    print('Code generated:\n', data.get('code'))
    stl_bytes = base64.b64decode(data['data'])
    with open('server_compiled.stl', 'wb') as f:
        f.write(stl_bytes)
    m = trimesh.load('server_compiled.stl')
    print('Extents:', m.extents)
    print('Bounds:\n', m.bounds)
else:
    print('Error:', r.text)
