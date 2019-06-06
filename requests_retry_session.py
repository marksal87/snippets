def requests_retry_session(retries=4, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None, ):
    """ Returns a requests session that can be used to query a URL with automatic retrying
    Usage: response = requests_retry_session().get('http://httpbin.org/delay/10')
       or: s = requests.Session()
           s.auth = ('user', 'pass')
           s.headers.update({'x-test': 'true'})
           response = requests_retry_session(session=s).get('https://www.peterbe.com')

    Credits: Peter Bengtsson
             https://www.peterbe.com/plog/best-practice-with-retries-with-requests
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
