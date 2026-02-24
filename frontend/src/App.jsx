import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import JobList from './pages/JobList';
import JobDetail from './pages/JobDetail';
import CreateJob from './pages/CreateJob';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Navbar />
        <main className="max-w-5xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/jobs" element={<JobList />} />
            <Route path="/jobs/:id" element={<JobDetail />} />
            <Route path="/jobs/new" element={<CreateJob />} />
          </Routes>
        </main>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;