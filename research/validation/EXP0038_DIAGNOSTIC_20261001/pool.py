import sys,glob,numpy as np
for name,root in (('parent','/tmp/diag35'),('assigned','/tmp/diag38/assigned'),('control','/tmp/diag38/control')):
    c=[];p=[];succ=[]
    for f in glob.glob(root+'/seed_*/ep_*.npz'):
        z=np.load(f); succ.append(bool(z['success']))
        mass,geo,act,route,assign,active=(z[k] for k in ('mass','geo','act','route','assign','active'))
        k=mass.shape[1]; geo=np.where(geo<0,np.inf,geo); a=np.clip(assign,0,k-1); alive=(mass>0).sum(1)
        ra=np.take_along_axis(route,a[:,:,None,None].repeat(3,3),axis=2)[:,:,0]
        ga=np.take_along_axis(geo,a[:,:,None],axis=2)[:,:,0]; gn=np.vstack([ga[1:],ga[-1:]])
        same=np.concatenate([alive[1:],alive[-1:]])==alive
        m=active&(assign>=0)&np.isfinite(ga)&np.isfinite(gn)&(ga>.7)&same[:,None]&(np.linalg.norm(act,axis=2)>1e-9)
        cos=(act*ra).sum(2)/np.maximum(np.linalg.norm(act,axis=2)*np.linalg.norm(ra,axis=2),1e-9)
        c.append(cos[m]); p.append(((ga-gn)/.1)[m])
    c=np.concatenate(c);p=np.concatenate(p)
    print(name,'episodes',len(succ),'replay success',round(np.mean(succ),3),'nav cos',round(c.mean(),3),'nav progress mm/s',round(p.mean(),3))
