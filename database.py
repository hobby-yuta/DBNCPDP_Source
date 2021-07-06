import re
import config
import MySQLdb
from datetime import timezone
import datetime
from github.GithubException import RateLimitExceededException,IncompletableObject
import time
from tqdm import tqdm
from github import Github

class Database:
    def __init__(self):
        self.conn = None
        self.reset()
    def __del__(self):
        self.conn.close()

    def reset(self):
        if not (self.conn is None):
            self.conn.close()

        self.conf = config.dbconf
        self.isError = False
        self.token = config.gitconf['keys']['token']
        self.conn = MySQLdb.connect(
            user =self.conf['user'],
            passwd =self.conf['passwd'],
            db =self.conf['db'],
            host =self.conf['host'],
            use_unicode=True,
            charset='utf8',
            autocommit=False
        )
        self.cur = self.conn.cursor()
        self.github = Github(self.token)
    def commit(self):
        if not (self.isError == True):
            print("committed")
            self.conn.commit()

    def __execute(self,query,format=None):
        result = []
        try:
            self.cur.execute(query,format)
            result = self.cur.fetchall()
        except MySQLdb.Error as e:
            self.isError = True
            print('error in Database.__execute(): ', e)
        return result

    def __executemany(self,query,formats=None):
        result = []
        try:
            self.cur.executemany(query,formats)
            result = self.cur.fetchall()
        except MySQLdb.Error as e:
            self.isError = True
            print(type(e),'in Database.__executemany(): ', e)
            print(formats)
        return result

    def __getColumnsDeclare(self,columns):
        result = []
        for key in columns.keys():
            result.append("{} {}".format(key,columns[key]))
        return ','.join(result)


    def createTable(self,tablename):
        table = self.conf['tables'][tablename]
        cols = table['columns']
        return self.__execute("CREATE TABLE IF NOT EXISTS {} ({})"
            .format(tablename, self.__getColumnsDeclare(table['columns'])))

    def searchCommits(self,rcnf,tagnames,tagdates):

        index = 0
        insert_query = "INSERT INTO commits VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

        """
        for _repo in self.github.get_organization(rcnf['organization']).get_repos(type='public'):
            if _repo.name == rcnf['repository']:
                repo = _repo
                break
        if not repo:
            print("repository({}) is not exist.".format(rcnf['repository']))
            return
        """
        try:
            repo = self.github.get_repo("{}/{}".format(rcnf['organization'],rcnf['repository']))
            #print(str(self.github.get_rate_limit().core))
            for i,_ in enumerate(tagnames):
                print(tagnames[i]," {}/{}".format(i,len(tagnames)))
                until = tagdates[i]
                since = tagdates[i+1] if i+1 < len(tagdates) else datetime.datetime(1900,1,1)
                params = []
                for _commit in repo.get_commits(since=since,until=until):
                    #limit = self.github.get_rate_limit()
                    #print(str(limit.core))
                    for file in _commit.files:
                        param = [
                        rcnf['organization'],
                        rcnf['repository'],
                        tagnames[i],
                        _commit.comments_url,
                        _commit.commit.message,
                        _commit.commit.url,
                        file.filename,
                        str(file.additions),
                        str(file.deletions),
                        str(file.changes),
                        file.blob_url,
                        file.patch,
                        file.previous_filename,
                        file.raw_url,
                        file.sha,
                        _commit.html_url,
                        _commit.parents[0].html_url,
                        _commit.sha,
                        _commit.committer.html_url if _commit.committer else "",
                        _commit.author.html_url if _commit.author else ""]
                        param = [p if p else "" for p in param]
                        params.append(param)

                if len(params) > 0:
                    self.__executemany(insert_query,params)
                    self.commit()
                index += 1
        except RateLimitExceededException as e:
            print('RateLimitExceededException is occured')
            print('index',index,tagnames[index])
            self.sleep_until_limit()
            self.searchCommits(rcnf,tagnames[index-1:],tagdates[index-1:])

        except IncompletableObject as e:
            print('IncompletableObject is occured')
            print('index',index,tagnames[index])
            self.searchCommits(rcnf,tagnames[index-1:],tagdates[index-1:])


    def searchTags(self,rcnf):
        try:
            repo = self.github.get_repo("{}/{}".format(rcnf['organization'],rcnf['repository']))
            tagnames = []
            tagdates = []
            for tag in repo.get_tags():
                tagnames.append(tag.name)
                tagdates.append(tag.commit.commit.author.date)
        except RateLimitExceededException as e:
            print('RateLimitExceededException is occured')
            self.sleep_until_limit()
            tagnames,tagdates = self.searchTags(rcnf)
        return tagnames,tagdates

    def sleep_until_limit(self,margin=60):
        limit = self.github.get_rate_limit()
        result = re.findall(r'Rate\(reset=(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+), remaining=\d+, limit=\d+\)',str(limit.core))
        result = [int(r) for r in result[0]]
        limit_datetime = datetime.datetime(result[0],result[1],result[2],result[3],result[4],result[5],tzinfo=timezone.utc)
        td = limit_datetime - datetime.datetime.now(timezone.utc)
        now = datetime.datetime.now()
        print('sleep {} to {}'.format(now,now+td))
        time.sleep(td.seconds+margin)
        self.reset()

    def SELECT(self,table,columns,where=None):
        if columns == []:
            query = "SELECT * FROM " + table
        else:
            query = "SELECT " +','.join(columns)+" FROM " + table

        if not where is None:
            query += " WHERE " + ' OR '.join(where)
        print(query)
        return self.__execute(query)

if __name__ == '__main__':
    db = Database()
    #db.createTable('commits')
    conf = config.gitconf['ant']
    since_tag = 'rel/1.7.1'
    until_tag = 'rel/1.5'

    names,dates = db.searchTags(conf)


    begin= -1
    end = -1
    for i,_ in enumerate(names):
        if names[i] == since_tag:
            begin = i
        if names[i] == until_tag:
            end = i
            break
    if begin == -1 or end == -1:
        exit()
    dates = dates[index:end]
    names = names[index:end]

    """
    columns =["commitMessage","fileFilename"]
    for i,t in enumerate(db.SELECT("commits",columns)):
        if i > 20:
            break
        print(t[0])
        print(t[1]+"\n\n")
    """
    db.searchCommits(conf,names,dates)
    db.commit()
    db.reset()
