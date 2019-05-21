from pyspark import SparkContext
import sys
from datetime import datetime

import fiona.crs
import geopandas as gpd

def distributeIndexFile(sc, zones, fn):
    import rtree
    index = rtree.Rtree(fn)
    for idx, geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    index.close()
    sc.addFile(fn + '.idx')
    sc.addFile(fn + '.dat')
    
def readIndexFile(fn):
    import rtree
    return rtree.Rtree(fn)

def createIndex(zones):
    import rtree
    index = rtree.Rtree()
    for idx, geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return index

def findZone(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None

def processTweets(pid, records):
    import csv
    import pyproj
    import shapely.geometry as geom
    
    reader = csv.reader(records, delimiter='|')
    proj = pyproj.Proj(init='epsg:2263', preserve_units=True)
    
    ctracts = CTRACTS.value
    index = createIndex(ctracts)
    
    drug_words = DRUG_WORDS.value
    
    counts = {}
    
    for row in reader:
        if any(word in row[5] for word in drug_words):
            try:
                point = geom.Point(proj(float(row[2]),
                                        float(row[1])))
            except:
                continue
            ct = findZone(point, index, ctracts)

            if ct:
                match = (ctracts.plctract10[ct], int(ctracts.plctrpop10[ct])) 
                counts[match] = counts.get(match, 0) + 1
    
    return counts.items()

if __name__=='__main__':
    
    start = datetime.now()
    
    sc = SparkContext()
    
    fn = sys.argv[1]
    out = sys.argv[2]
    
    ctracts = gpd.read_file('500cities_tracts.geojson').to_crs(fiona.crs.from_epsg(2263))
    
    #There are some invalid polygons, I'll get rid of them to avoid error when intersecting
    #Also get rid of census tracts with 0 population
    ctracts = ctracts.loc[(ctracts.plctrpop10.apply(lambda x: bool(int(x)))) & \
                          (ctracts.geometry.apply(lambda x: x.is_valid)),:].reset_index(drop=True)

    CTRACTS = sc.broadcast(ctracts)
    
    drug_words = {line.strip() for line in open('drug_illegal.txt')} | {line.strip() for line in open('drug_sched2.txt')}
    DRUG_WORDS = sc.broadcast(drug_words)

    tweets = sc.textFile(fn, use_unicode=True)
    
    result = tweets.mapPartitionsWithIndex(processTweets) \
            .reduceByKey(lambda x,y: x+y) \
            .sortByKey() \
            .map(lambda x: ','.join([x[0][0], str(x[1]/x[0][1])])) \
            .coalesce(1)
    
    result.saveAsTextFile(out)
    
    print('Done! Total time: {} minutes'.format((datetime.now()-start).total_seconds()/60))