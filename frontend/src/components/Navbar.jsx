import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard' },
    { path: '/jobs', label: 'Jobs' },
    { path: '/jobs/new', label: 'New Job' },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-white border-b shadow-sm">
      <div className="max-w-5xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="text-xl font-bold text-gray-800">
            ML Task Engine
          </Link>

          <div className="flex space-x-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  isActive(item.path)
                    ? 'bg-blue-100 text-blue-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}