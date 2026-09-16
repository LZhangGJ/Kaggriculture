"""Portable CPU tree inference; accepts only the legal observation features."""
import struct


def predict(model,features):
    if 'ensemble' in model:
        result=[0.0]*len(model['names'])
        for component in model['ensemble']:
            scores=predict(component['model'],features)
            for i,value in enumerate(scores):result[i]+=component['weight']*value
        return result
    if model.get('economic_features') and any(name not in features for name in model['features']):
        from features import economic_plan_features
        features=dict(features,**economic_plan_features(features,model['profiles']))
    if 'action_inputs' in model:
        scalar=model['scalar_model']
        return [predict(scalar,{key:features[source] if isinstance(source,str) else source
                                for key,source in model['action_inputs'][name].items()})[0]
                for name in model['names']]
    # sklearn tree traversal uses float32 inputs, even when fit receives float64.
    values=[features[name] for name in model['features']]
    if model.get('input_dtype','float32')=='float32':values=[struct.unpack('f',struct.pack('f',v))[0] for v in values]
    result=[0.0]*len(model['names'])
    for tree in model['trees']:
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if values[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        if 'output_index' in tree:
            result[tree['output_index']]+=tree['value'][node][0]*tree.get('scale',1.)
        else:
            for i,value in enumerate(tree['value'][node]):result[i]+=value
    return [v/model.get('divisor',len(model['trees'])) for v in result]


def choose(model,features):
    scores=predict(model,features);preferred=None
    if model.get('target')=='continuation_advantage':
        matches=[]
        for i,name in enumerate(model['names']):
            cfg=dict(model['base_config'],**model['profiles'][name])
            if all(abs(features['current_setting_'+k]-v)<1e-10 for k,v in cfg.items()):matches.append(i)
        if len(matches)!=1:raise ValueError('Continuation profile is not uniquely represented')
        # Continuing the current policy has exactly zero advantage by definition.
        scores[matches[0]]=0.0
        preferred=matches[0]
    index=preferred if preferred is not None and max(scores)<=model.get('minimum_advantage',0.0) else max(range(len(scores)),key=scores.__getitem__)
    name=model['names'][index]
    return name,model['profiles'][name],scores
